#include "test/runner/Pipelines.h"
#include "src/conversion/LowerTensorToGPU.h"
#include "src/conversion/LowerTensorToSCF.h"
#include "src/conversion/MetalExporter.h"
#include "src/tensor/GPURunner.h"
#include "src/tensor/MetalRuntime.h"

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/Error.h"

#include <algorithm>

namespace sconeml::test {
namespace {
std::string print(mlir::ModuleOp module) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  module.print(stream);
  return text;
}

StridedMemRefType<int32_t, 1> descriptor(std::vector<int32_t> &storage,
                                       std::size_t count) {
  return {storage.data(), storage.data() + 1, 0,
          {static_cast<int64_t>(count)}, {1}};
}
} // namespace

void executeMapIR(mlir::ModuleOp module, const TestCase &test, const Run &run,
                  Output &output, bool disableMetal) {
  output.stage = "verify";
  if (mlir::failed(mlir::verify(module)))
    return;
  // These are test-adapter preconditions, not expected compiler failures.
  // Restrict the host wrapper so no unrelated operation can affect the result.
  output.stage = "runner";
  auto functions = module.getOps<mlir::func::FuncOp>();
  require(llvm::hasSingleElement(module.getBody()->getOperations()) &&
              llvm::hasSingleElement(functions),
          "IR execution requires exactly one func.func");
  auto function = *functions.begin();
  require(llvm::hasSingleElement(function.getBody()) &&
              function.getNumArguments() == 2 &&
              function.getFunctionType().getNumResults() == 0,
          "IR execution requires one block, two buffers and no function results");
  auto &body = function.getBody().front();
  require(body.getOperations().size() == 2 &&
              mlir::isa<letalg::TensorMapOp>(body.front()) &&
              mlir::isa<mlir::func::ReturnOp>(body.back()),
          "IR execution requires only a tensor_map followed by func.return");
  auto map = mlir::cast<letalg::TensorMapOp>(body.front());
  auto type = mlir::cast<mlir::MemRefType>(map.getInput().getType());
  require(type.hasStaticShape() && type.getLayout().isIdentity() &&
              !type.getMemorySpace() &&
              map.getInput() == function.getArgument(0) &&
              map.getOutput() == function.getArgument(1),
          "IR execution requires static identity-layout host buffer arguments");
  require(test.tensorInput &&
              static_cast<uint64_t>(type.getDimSize(0)) == test.tensorInput->size(),
          "tensor-input length must match the static map shape");
  const auto &input = *test.tensorInput;

  if (run.pipeline == "tensor-metal-ir") {
    auto before = print(module);
    output.stage = "gpu-lowering";
    auto device = sconeml::lowerTensorMapToGPU(map);
    if (mlir::failed(device))
      return;
    output.ir = print(**device);
    output.stage = "msl-export";
    auto kernel = sconeml::exportMetalKernel(**device);
    if (mlir::failed(kernel))
      return;
    output.sourceUnchanged = before == print(module) && output.ir == print(**device);
    output.text = kernel->source;
    output.entrypoint = kernel->entrypoint;
    output.stage = "metal-execute";
    // Empty execution is a runtime no-op, so it can be tested without a GPU.
    if (!input.empty() && (disableMetal || !tensor::gpuIsAvailable())) {
      output.skipped = true;
      output.diagnostic = "Metal device unavailable or disabled";
      return;
    }
    std::vector<int32_t> result(input.size(), 0x13579bdf);
    tensor::runMetalKernel(kernel->source, kernel->entrypoint, input, result,
                           input.size(), {run.threadsPerThreadgroup.value_or(0)});
    output.number = Number{true, std::move(result)};
  } else {
    output.stage = "cpu-lowering";
    function->setAttr("llvm.emit_c_interface",
                      mlir::UnitAttr::get(module.getContext()));
    const auto entrypoint = "_mlir_ciface_" + function.getSymName().str();
    mlir::PassManager passes(module.getContext());
    passes.addPass(sconeml::createLowerTensorToSCFPass());
    passes.addPass(mlir::createSCFToControlFlowPass());
    passes.addPass(mlir::createConvertControlFlowToLLVMPass());
    passes.addPass(mlir::createArithToLLVMConversionPass());
    passes.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
    passes.addPass(mlir::createConvertFuncToLLVMPass());
    passes.addPass(mlir::createReconcileUnrealizedCastsPass());
    if (mlir::failed(passes.run(module)) || mlir::failed(mlir::verify(module)))
      return;
    output.ir = print(module);
    output.stage = "cpu-execute";
    auto engine = mlir::ExecutionEngine::create(module);
    if (!engine)
      throw std::runtime_error(llvm::toString(engine.takeError()));
    auto address = (*engine)->lookup(entrypoint);
    if (!address)
      throw std::runtime_error(llvm::toString(address.takeError()));
    // Poison results so a missing store cannot silently look like a zero.
    constexpr int32_t sentinel = 0x13579bdf;
    std::vector<int32_t> inputStorage(input.size() + 2, sentinel);
    std::vector<int32_t> outputStorage(input.size() + 2, sentinel);
    std::copy(input.begin(), input.end(), inputStorage.begin() + 1);
    const auto inputBefore = inputStorage;
    auto in = descriptor(inputStorage, input.size());
    auto out = descriptor(outputStorage, input.size());
    using Function = void (*)(StridedMemRefType<int32_t, 1> *,
                              StridedMemRefType<int32_t, 1> *);
    reinterpret_cast<Function>(*address)(&in, &out);
    require(inputStorage == inputBefore, "CPU map modified its input");
    require(outputStorage.front() == sentinel && outputStorage.back() == sentinel,
            "CPU map wrote outside its output range");
    output.number = Number{true, {outputStorage.begin() + 1, outputStorage.end() - 1}};
  }
  output.success = true;
}
} // namespace sconeml::test
