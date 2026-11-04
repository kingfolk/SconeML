#include "src/conversion/MetalExporter.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <stdexcept>
#include <string>

namespace {
void require(bool condition, const std::string &message) {
  if (!condition)
    throw std::runtime_error(message);
}

std::string readFile(const char *path) {
  auto file = llvm::MemoryBuffer::getFile(path);
  require(bool(file), std::string("cannot read fixture: ") + path);
  return (*file)->getBuffer().str();
}

std::string print(mlir::ModuleOp module) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  module.print(stream);
  return text;
}

std::string replace(std::string text, const std::string &old,
                    const std::string &replacement) {
  size_t position = 0, count = 0;
  while ((position = text.find(old, position)) != std::string::npos) {
    text.replace(position, old.size(), replacement);
    position += replacement.size();
    ++count;
  }
  require(count != 0, "negative test did not modify its fixture");
  return text;
}

void expectFailure(mlir::MLIRContext &context, const std::string &text,
                   const std::string &reason, const std::string &opName) {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(text, &context);
  require(bool(module), "invalid negative-test setup: " + reason);
  std::string diagnostic;
  mlir::ScopedDiagnosticHandler handler(&context, [&](mlir::Diagnostic &diag) {
    llvm::raw_string_ostream stream(diagnostic);
    diag.print(stream);
    return mlir::success();
  });
  auto result = sconeml::exportMetalKernel(*module);
  require(mlir::failed(result), "unsupported IR returned executable MSL");
  require(diagnostic.find("MSL export:") != std::string::npos &&
              diagnostic.find(reason) != std::string::npos &&
              diagnostic.find(opName) != std::string::npos,
          "missing stage/op/reason in diagnostic: " + diagnostic);
}

void testRejections(mlir::MLIRContext &context, const std::string &text) {
  expectFailure(context, "module {}", "exactly one gpu.module",
                "builtin.module");
  expectFailure(context,
                replace(text, "gpu.module @kernels {",
                        "gpu.module @kernels {\n"
                        "    gpu.func @other() kernel { gpu.return }"),
                "exactly one defined GPU kernel", "gpu.module");
  expectFailure(context,
                replace(replace(text, "%count: i32", "%count: i64"),
                        "i32 to index", "i64 to index"),
                "an i32 count", "gpu.func");
  expectFailure(context,
                replace(text, "arith.muli %x, %x", "arith.divui %x, %x"),
                "unsupported operation", "arith.divui");
  expectFailure(context,
                replace(text, "arith.index_castui", "arith.index_cast"),
                "unsupported operation", "arith.index_cast");
  expectFailure(context, replace(text, "arith.cmpi ult", "arith.cmpi ule"),
                "unsigned count cast/comparison", "gpu.func");
  expectFailure(context, replace(text, "gpu.block_id x", "gpu.block_id y"),
                "x-dimensional", "gpu.func");
  expectFailure(context,
                replace(text, "#gpu.address_space<global>",
                        "#gpu.address_space<workgroup>"),
                "global identity-layout", "gpu.func");
  expectFailure(
      context,
      replace(text, "memref<?xi32,", "memref<?xi32, strided<[2], offset: 1>,"),
      "identity-layout", "gpu.func");
  expectFailure(context,
                replace(text, "arith.muli %x, %x : i32",
                        "arith.muli %x, %x overflow<nsw> : i32"),
                "no-wrap promises", "arith.muli");
  expectFailure(context,
                replace(text, "arith.muli %x, %x", "arith.muli %x, %count"),
                "not a local element", "arith.muli");
  expectFailure(context,
                replace(text, "memref.store %twice, %output",
                        "memref.store %twice, %input"),
                "output[index] store", "scf.if");
  expectFailure(context,
                replace(text, "%square = arith.muli",
                        "%unused = arith.constant 1.0 : f32\n"
                        "        %square = arith.muli"),
                "only i32 constants", "arith.constant");
  expectFailure(context,
                replace(text, "scf.if %in_bounds {",
                        "%extra = arith.constant 0 : index\n"
                        "      scf.if %in_bounds {"),
                "map index/guard/return envelope", "gpu.func");
}

void testInvalidIR(mlir::MLIRContext &context, const std::string &text) {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(text, &context);
  require(bool(module), "cannot parse invalid-IR test setup");
  mlir::memref::StoreOp store;
  module->walk([&](mlir::memref::StoreOp op) { store = op; });
  mlir::OpBuilder builder(store);
  auto wrong =
      mlir::arith::ConstantIntOp::create(builder, store.getLoc(), 1, 64);
  store->setOperand(0, wrong.getResult());
  std::string diagnostic;
  mlir::ScopedDiagnosticHandler handler(&context, [&](mlir::Diagnostic &diag) {
    llvm::raw_string_ostream stream(diagnostic);
    diag.print(stream);
    return mlir::success();
  });
  auto result = sconeml::exportMetalKernel(*module);
  require(mlir::failed(result) &&
              diagnostic.find("MSL export: input IR verification failed") !=
                  std::string::npos,
          "malformed IR must fail before MSL generation");
}

void writeSource(llvm::StringRef directory, llvm::StringRef filename,
                 const std::string &source) {
  llvm::SmallString<256> path(directory);
  llvm::sys::path::append(path, filename);
  std::error_code error;
  llvm::raw_fd_ostream stream(path, error);
  require(!error, "cannot write generated MSL: " + error.message());
  stream << source;
  stream.close();
  require(!stream.has_error(), "failed to write complete generated MSL");
}
} // namespace

int main(int argc, char **argv) {
  try {
    require(argc == 5, "usage: metal_exporter <gpu.mlir> <expected.metal> "
                       "<integer_bits.mlir> <output-directory>");
    mlir::DialectRegistry registry;
    registry.insert<mlir::arith::ArithDialect, mlir::gpu::GPUDialect,
                    mlir::memref::MemRefDialect, mlir::scf::SCFDialect>();
    mlir::MLIRContext context(registry);
    context.loadAllAvailableDialects();
    auto text = readFile(argv[1]);
    auto module = mlir::parseSourceString<mlir::ModuleOp>(text, &context);
    require(bool(module), "cannot parse GPU fixture");
    auto before = print(*module);
    auto result = sconeml::exportMetalKernel(*module);
    require(mlir::succeeded(result), "valid GPU IR rejected");
    require(result->entrypoint == "tensor_map", "wrong entrypoint");
    require(result->source == readFile(argv[2]),
            "MSL differs from golden fixture");
    require(before == print(*module), "export modified the input IR");
    // The golden file requires exactly one square definition and references
    // the same temporary twice in the addition, not two expanded expressions.
    testRejections(context, text);
    testInvalidIR(context, text);

    auto integer =
        mlir::parseSourceString<mlir::ModuleOp>(readFile(argv[3]), &context);
    require(bool(integer), "cannot parse integer-bit fixture");
    auto bits = sconeml::exportMetalKernel(*integer);
    require(mlir::succeeded(bits), "integer-bit fixture rejected");
    for (auto literal : {"4294967295u", "2147483648u", "2147483647u", "0u"})
      require(bits->source.find(literal) != std::string::npos,
              std::string("missing unsigned bit pattern: ") + literal);
    require(bits->source.find("const int ") == std::string::npos,
            "signed arithmetic could change overflow semantics");
    auto error = llvm::sys::fs::create_directories(argv[4]);
    require(!error, "cannot create MSL output directory: " + error.message());
    writeSource(argv[4], "shared.metal", result->source);
    writeSource(argv[4], "integer.metal", bits->source);
    llvm::outs() << "PASS MSL golden output, SSA reuse, uint bits and "
                    "rejection diagnostics\n";
    return 0;
  } catch (const std::exception &error) {
    llvm::errs() << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
