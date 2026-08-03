#include "src/tensor/CPURunner.h"

#include "src/conversion/LowerTensorToSCF.h"
#include "src/tensor/TensorIR.h"

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/TargetSelect.h"

#include <memory>
#include <stdexcept>

namespace sconeml::tensor {
namespace {

[[noreturn]] void throwLLVMError(llvm::Error error,
                                 const std::string &message) {
  throw std::runtime_error(message + ": " + llvm::toString(std::move(error)));
}

void lowerToLLVM(mlir::ModuleOp module) {
  mlir::PassManager passes(module.getContext());
  passes.addPass(sconeml::createLowerTensorToSCFPass());
  passes.addPass(mlir::createSCFToControlFlowPass());
  passes.addPass(mlir::createConvertControlFlowToLLVMPass());
  passes.addPass(mlir::createArithToLLVMConversionPass());
  passes.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
  passes.addPass(mlir::createConvertFuncToLLVMPass());
  passes.addPass(mlir::createReconcileUnrealizedCastsPass());
  if (mlir::failed(passes.run(module)))
    throw std::runtime_error("tensor CPU lowering to LLVM failed");
  if (mlir::failed(mlir::verify(module)))
    throw std::runtime_error("lowered tensor LLVM module failed verification");
}

} // namespace

std::vector<float> runPolynomialCPU(const std::vector<float> &input,
                                    std::string *loweredLLVMIR) {
  static const bool initialized = [] {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    return true;
  }();
  (void)initialized;

  TensorModule tensorModule = buildPolynomialModule();
  lowerToLLVM(*tensorModule.module);
  if (loweredLLVMIR)
    *loweredLLVMIR = printModule(tensorModule.module.get());

  mlir::ExecutionEngineOptions options;
  auto transformer = mlir::makeOptimizingTransformer(2, 0, nullptr);
  options.transformer = transformer;
  auto maybeEngine =
      mlir::ExecutionEngine::create(tensorModule.module.get(), options);
  if (!maybeEngine)
    throwLLVMError(maybeEngine.takeError(), "could not create CPU JIT");
  std::unique_ptr<mlir::ExecutionEngine> engine = std::move(*maybeEngine);

  std::vector<float> output(input.size(), 0.0f);
  StridedMemRefType<float, 1> inputDescriptor{
      const_cast<float *>(input.data()), const_cast<float *>(input.data()), 0,
      {static_cast<int64_t>(input.size())}, {1}};
  StridedMemRefType<float, 1> outputDescriptor{
      output.data(), output.data(), 0, {static_cast<int64_t>(output.size())},
      {1}};
  auto maybeFunction = engine->lookup("_mlir_ciface_tensor_polynomial");
  if (!maybeFunction)
    throwLLVMError(maybeFunction.takeError(), "CPU JIT lookup failed");
  using TensorFunction = void (*)(StridedMemRefType<float, 1> *,
                                  StridedMemRefType<float, 1> *);
  auto function = reinterpret_cast<TensorFunction>(*maybeFunction);
  function(&inputDescriptor, &outputDescriptor);
  return output;
}

} // namespace sconeml::tensor
