#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "src/tensor/CPURunner.h"
#include "src/tensor/GPURunner.h"
#include "src/tensor/Sum.h"
#include "src/tensor/SumGPU.h"
#include <array>
#include <iostream>
#include <stdexcept>

int main(int argc, char **argv) {
  try {
    bool metal = false, ir = false, msl = false;
    for (int i = 1; i < argc; ++i) {
      std::string flag = argv[i];
      if (flag == "--metal")
        metal = true;
      else if (flag == "--ir")
        ir = true;
      else if (flag == "--msl")
        msl = true;
      else
        throw std::invalid_argument(
            "usage: sum_walkthrough [--metal] [--ir] [--msl]");
    }
    using namespace sconeml::tensor;
    std::vector<float> a{1, 2, 3, 4, 5, 6}, b{10, 20, 30};
    std::array<ShapedTensorF32Argument, 2> args{
        {{"a", a, {2, 3}}, {"b", b, {3}}}};
    auto mapped = metal ? runShapedTensorF32GPU("a * b", args)
                        : runShapedTensorF32CPU("a * b", args);
    auto result = metal ? runSumF32GPU("sum(a * b, axis=1)", args)
                        : runSumF32CPU("sum(a * b, axis=1)", args);
    if (mapped.values != std::vector<float>({10, 40, 90, 40, 100, 180}) ||
        mapped.shape != std::vector<int64_t>({2, 3}) ||
        result.values != std::vector<float>({140, 320}) ||
        result.shape != std::vector<int64_t>({2}))
      throw std::runtime_error(
          "walkthrough numerical/shape expectation failed");
    std::cout << "Backend: " << (metal ? gpuName() : "CPU LLVM JIT")
              << "\nSource: sum(a * b, axis=1)\n";
    std::cout << "a[2,3] = [[1,2,3],[4,5,6]], b[3] = [10,20,30]\n";
    std::cout << "broadcast map[2,3] = [[10,40,90],[40,100,180]]\n";
    std::cout << "owned sum[2] = [140,320]\n";
    auto plan = makeSumPlan(std::array<int64_t, 2>{2, 3}, 1);
    auto program = exportSumMetal(plan, true);
    for (size_t i = 0; i < program.dispatches.size(); ++i) {
      const auto &d = program.dispatches[i];
      std::cout << "sum pass " << i << ": length=" << d.length
                << ", parts/output=" << d.parts << ", groups=" << d.groupCount
                << ", width=256\n";
    }
    auto large =
        exportSumMetal(makeSumPlan(std::array<int64_t, 1>{65537}, {}), true);
    std::cout << "large global plan (65537 inputs): ";
    for (const auto &d : large.dispatches)
      std::cout << d.length << " -> ";
    std::cout << "1 (three dispatches; plan only)\n";
    std::cout.flush();
    if (ir) {
      mlir::MLIRContext context;
      context.getOrLoadDialect<mlir::arith::ArithDialect>();
      context.getOrLoadDialect<mlir::func::FuncDialect>();
      context.getOrLoadDialect<mlir::memref::MemRefDialect>();
      context.getOrLoadDialect<mlir::linalg::LinalgDialect>();
      auto mapModule = mlir::OwningOpRef<mlir::ModuleOp>(
          mlir::ModuleOp::create(mlir::UnknownLoc::get(&context)));
      std::array<ShapedTensorArgumentSpec, 2> specs{
          {{"a", 6, {2, 3}}, {"b", 3, {3}}}};
      translateShapedTensorF32Expression(*mapModule, "a * b", specs);
      auto sumModule = mlir::OwningOpRef<mlir::ModuleOp>(
          mlir::ModuleOp::create(mlir::UnknownLoc::get(&context)));
      buildSumCPU(*sumModule, plan, true);
      llvm::outs() << "\n// MAP LINALG IR (separate compilation unit)\n";
      mapModule->print(llvm::outs());
      llvm::outs() << "\n// SUM TREE LINALG IR (separate compilation unit)\n";
      sumModule->print(llvm::outs());
      llvm::outs() << "\n";
      llvm::outs().flush();
    }
    if (msl)
      std::cout << "\n// SUM MSL\n" << program.source;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
