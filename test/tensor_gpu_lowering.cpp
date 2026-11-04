#include "src/conversion/LowerTensorToGPU.h"
#include "src/conversion/MetalExporter.h"
#include "src/tensor/TensorFrontend.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace {

void require(bool condition, const std::string &message) {
  if (!condition)
    throw std::runtime_error(message);
}

bool equivalent(mlir::ModuleOp lhs, mlir::ModuleOp rhs) {
  mlir::IRMapping values;
  return mlir::OperationEquivalence::isEquivalentTo(
      lhs, rhs,
      [&](mlir::Value a, mlir::Value b) {
        return mlir::success(values.lookupOrNull(a) == b);
      },
      [&](mlir::Value a, mlir::Value b) { values.map(a, b); },
      mlir::OperationEquivalence::IgnoreLocations);
}

sconeml::letalg::TensorMapOp findMap(mlir::ModuleOp module) {
  sconeml::letalg::TensorMapOp map;
  unsigned count = 0;
  module.walk([&](sconeml::letalg::TensorMapOp candidate) {
    map = candidate;
    ++count;
  });
  require(count == 1, "test input must contain exactly one map");
  return map;
}

mlir::OwningOpRef<mlir::ModuleOp> lower(mlir::ModuleOp source) {
  mlir::OwningOpRef<mlir::ModuleOp> snapshot = source.clone();
  auto result = sconeml::lowerTensorMapToGPU(findMap(source));
  require(mlir::succeeded(result), "GPU lowering failed");
  auto device = std::move(*result);
  require(mlir::succeeded(mlir::verify(*device)), "invalid GPU output");
  auto msl = sconeml::exportMetalKernel(*device);
  require(mlir::succeeded(msl), "lowered map was rejected by the MSL exporter");
  require(equivalent(source, *snapshot), "extraction changed the host IR");
  device->walk([](mlir::Operation *op) {
    require(op->getName().getDialectNamespace() != "letalg",
            "device IR contains an unlowered language operation");
    require(!mlir::isa<mlir::func::FuncOp, mlir::memref::AllocOp>(op),
            "host operations leaked into the device module");
  });
  return device;
}

void expectFailure(mlir::ModuleOp source, const std::string &reason) {
  std::string diagnostic;
  mlir::ScopedDiagnosticHandler handler(
      source.getContext(), [&](mlir::Diagnostic &diag) {
        llvm::raw_string_ostream stream(diagnostic);
        diag.print(stream);
        return mlir::success();
      });
  auto result = sconeml::lowerTensorMapToGPU(findMap(source));
  require(mlir::failed(result), "invalid input produced a device module");
  require(diagnostic.find(reason) != std::string::npos,
          "expected diagnostic '" + reason + "', got: " + diagnostic);
}

void testSharing(mlir::ModuleOp module) {
  mlir::arith::MulIOp square;
  mlir::arith::AddIOp twice;
  module.walk([&](mlir::arith::MulIOp op) {
    if (op.getType().isInteger(32)) {
      require(!square, "scalar square was duplicated");
      square = op;
    }
  });
  module.walk([&](mlir::arith::AddIOp op) {
    if (op.getType().isInteger(32))
      twice = op;
  });
  require(square && twice, "missing shared scalar computation");
  require(twice.getLhs() == square.getResult() &&
              twice.getRhs() == square.getResult(),
          "SSA sharing was lost during extraction");
}

void testInvalidSources(mlir::ModuleOp source) {
  {
    mlir::OwningOpRef<mlir::ModuleOp> invalid = source.clone();
    auto map = findMap(*invalid);
    mlir::OpBuilder builder(map);
    auto external =
        mlir::arith::ConstantIntOp::create(builder, map.getLoc(), 2, 32);
    auto &square = map.getBody().front().front();
    square.setOperand(1, external.getResult());
    expectFailure(*invalid, "capture");
  }
  {
    mlir::OwningOpRef<mlir::ModuleOp> invalid = source.clone();
    auto map = findMap(*invalid);
    auto &square = map.getBody().front().front();
    mlir::OpBuilder builder(&square);
    auto unsupported = mlir::arith::XOrIOp::create(
        builder, square.getLoc(), square.getOperand(0), square.getOperand(1));
    square.getResult(0).replaceAllUsesWith(unsupported.getResult());
    square.erase();
    expectFailure(*invalid, "only i32 constants and add/sub/mul");
  }
  {
    mlir::OwningOpRef<mlir::ModuleOp> invalid = source.clone();
    auto map = findMap(*invalid);
    auto yield = mlir::cast<sconeml::letalg::YieldOp>(
        map.getBody().front().getTerminator());
    mlir::OpBuilder builder(yield);
    auto wrong =
        mlir::arith::ConstantIntOp::create(builder, map.getLoc(), 1, 64);
    yield->setOperand(0, wrong.getResult());
    expectFailure(*invalid, "body must yield");
  }
}

void testLayoutRejection(mlir::MLIRContext &context, const std::string &text) {
  for (const auto &type :
       {"memref<4xi32, strided<[2], offset: 1>>", "memref<4xi32, 3>"}) {
    auto changed = text;
    const std::string original = "memref<4xi32>";
    size_t position = 0;
    while ((position = changed.find(original, position)) != std::string::npos) {
      changed.replace(position, original.size(), type);
      position += std::string(type).size();
    }
    auto module = mlir::parseSourceString<mlir::ModuleOp>(changed, &context);
    require(bool(module), "could not parse layout rejection fixture");
    expectFailure(*module, "identity layout in the default host memory space");
  }
}

void testFrontend(mlir::MLIRContext &context) {
  for (const auto &source :
       {"let xs = [1, 2, 3] in xs * xs + 2 * xs + 1",
        "let xs = [0, 4, 7] in 10 - xs * 3", "let xs = [1] in xs"}) {
    mlir::OwningOpRef<mlir::ModuleOp> host =
        mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
    sconeml::tensor::translateTensorSource(*host, source);
    auto device = lower(*host);
    unsigned stores = 0;
    device->walk([&](mlir::memref::StoreOp) { ++stores; });
    require(stores == 1, "expected one guarded device store");
  }
}

} // namespace

int main(int argc, char **argv) {
  try {
    require(
        argc == 3 || (argc == 4 && std::string(argv[3]) == "--dump"),
        "usage: tensor_gpu_lowering <source.mlir> <expected.mlir> [--dump]");
    mlir::DialectRegistry registry;
    registry.insert<sconeml::letalg::LetAlgDialect, mlir::arith::ArithDialect,
                    mlir::func::FuncDialect, mlir::memref::MemRefDialect,
                    mlir::scf::SCFDialect>();
    mlir::MLIRContext context(registry);
    context.loadAllAvailableDialects();
    auto buffer = llvm::MemoryBuffer::getFile(argv[1]);
    require(bool(buffer), "could not read source fixture");
    auto text = (*buffer)->getBuffer().str();
    auto source = mlir::parseSourceString<mlir::ModuleOp>(text, &context);
    require(bool(source), "could not parse source fixture");
    // GPU dialect is loaded by the lowering API, not by the caller.
    auto device = lower(*source);
    auto expected = mlir::parseSourceFile<mlir::ModuleOp>(argv[2], &context);
    require(bool(expected), "could not parse expected device fixture");
    require(equivalent(*device, *expected), "device IR differs from fixture");
    testSharing(*device);

    std::string printed;
    llvm::raw_string_ostream stream(printed);
    device->print(stream);
    auto roundTrip = mlir::parseSourceString<mlir::ModuleOp>(printed, &context);
    require(roundTrip && equivalent(*device, *roundTrip),
            "IR round trip failed");
    if (argc == 4)
      llvm::outs() << printed << '\n';
    testInvalidSources(*source);
    testLayoutRejection(context, text);
    testFrontend(context);
    llvm::outs() << "PASS TensorMap GPU extraction, fixture, SSA sharing, "
                    "diagnostics and frontend integration (IR only)\n";
    return 0;
  } catch (const std::exception &error) {
    llvm::errs() << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
