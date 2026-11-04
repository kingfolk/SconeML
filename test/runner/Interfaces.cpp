#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "src/conversion/LowerTensorToGPU.h"
#include "src/conversion/MetalExporter.h"
#include "src/dialect/LetAlgDialect.h"
#include "src/tensor/TensorFrontend.h"
#include "test/runner/TestSpec.h"
#include "llvm/Support/raw_ostream.h"

// Two C++ interface contracts remain outside the data runner:
// 1. Frontend translation links without CPURunner or ExecutionEngine.
// 2. The exporter handles malformed in-memory IR that a text parser cannot
//    construct (an i64 store operand attached to an i32 memref operation).
// Program contents still come from the same self-contained data files.
int main(int argc, char **argv) {
  using namespace sconeml::test;
  try {
    require(argc == 3, "usage: tensor_interfaces <source.ml> <gpu.mlir>");
    mlir::DialectRegistry registry;
    registry.insert<sconeml::letalg::LetAlgDialect, mlir::arith::ArithDialect,
                    mlir::func::FuncDialect, mlir::memref::MemRefDialect,
                    mlir::scf::SCFDialect>();
    mlir::MLIRContext context(registry);
    context.loadAllAvailableDialects();
    mlir::OwningOpRef<mlir::ModuleOp> host =
        mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
    sconeml::tensor::translateTensorSource(*host, readCase(argv[1]).program);
    require(mlir::succeeded(mlir::verify(*host)), "standalone frontend failed");
    require(!context.getLoadedDialect<mlir::gpu::GPUDialect>(),
            "GPU dialect already loaded by caller");
    sconeml::letalg::TensorMapOp map;
    host->walk([&](sconeml::letalg::TensorMapOp op) { map = op; });
    auto extracted = sconeml::lowerTensorMapToGPU(map);
    require(mlir::succeeded(extracted) &&
                context.getLoadedDialect<mlir::gpu::GPUDialect>(),
            "GPU extraction must load its dependent dialect");
    auto gpu = mlir::parseSourceString<mlir::ModuleOp>(
        readCase(argv[2]).program, &context);
    require(bool(gpu), "cannot parse GPU input");
    mlir::memref::StoreOp store;
    gpu->walk([&](mlir::memref::StoreOp op) { store = op; });
    require(bool(store), "GPU fixture must contain a store");
    mlir::OpBuilder builder(store);
    auto wrong =
        mlir::arith::ConstantIntOp::create(builder, store.getLoc(), 1, 64);
    store->setOperand(0, wrong.getResult());
    std::string diagnostic;
    mlir::ScopedDiagnosticHandler handler(
        &context, [&](mlir::Diagnostic &diag) {
          llvm::raw_string_ostream stream(diagnostic);
          diag.print(stream);
          return mlir::success();
        });
    auto result = sconeml::exportMetalKernel(*gpu);
    require(mlir::failed(result) &&
                diagnostic.find("MSL export: input IR verification failed") !=
                    std::string::npos,
            "exporter accepted malformed in-memory IR");
    llvm::outs() << "PASS standalone frontend linkage and malformed-IR "
                    "interface contracts\n";
    return 0;
  } catch (const std::exception &error) {
    llvm::errs() << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
