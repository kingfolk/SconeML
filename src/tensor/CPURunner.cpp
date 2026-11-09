#include "src/tensor/CPURunner.h"
#include "src/tensor/TensorFrontend.h"
#include "src/dialect/LetAlgDialect.h"

#include "src/conversion/ClosureConversion.h"
#include "src/conversion/LowerTensorToSCF.h"
#include "src/conversion/LowerToLLVM.h"
#include "src/conversion/UnwrapLet.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <array>
#include <memory>
#include <stdexcept>

namespace sconeml::tensor {
namespace {

[[noreturn]] void throwLLVMError(llvm::Error error,
                                 const std::string &message) {
  throw std::runtime_error(message + ": " + llvm::toString(std::move(error)));
}

void lowerToLLVM(mlir::ModuleOp module, bool shaped) {
  if (shaped)
    validateShapedTensorExpression(module);
  mlir::PassManager passes(module.getContext());
  if (shaped) {
    passes.addNestedPass<mlir::func::FuncOp>(mlir::createConvertLinalgToLoopsPass());
    passes.addPass(mlir::createLowerAffinePass());
  }
  passes.addPass(sconeml::createUnwrapLetPass());
  passes.addPass(sconeml::createClosureConversionPass());
  passes.addPass(sconeml::createLowerTensorToSCFPass());
  passes.addPass(sconeml::createLowerToLLVMPass());
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

std::unique_ptr<mlir::MLIRContext> makeCPUContext() {
  static const bool initialized = [] {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    return true;
  }();
  (void)initialized;

  mlir::DialectRegistry registry;
  mlir::registerAllToLLVMIRTranslations(registry);
  auto context = std::make_unique<mlir::MLIRContext>(registry);
  context->getOrLoadDialect<sconeml::letalg::LetAlgDialect>();
  context->getOrLoadDialect<mlir::arith::ArithDialect>();
  context->getOrLoadDialect<mlir::linalg::LinalgDialect>();
  context->getOrLoadDialect<mlir::cf::ControlFlowDialect>();
  context->getOrLoadDialect<mlir::func::FuncDialect>();
  context->getOrLoadDialect<mlir::LLVM::LLVMDialect>();
  context->getOrLoadDialect<mlir::memref::MemRefDialect>();
  context->getOrLoadDialect<mlir::scf::SCFDialect>();

  return context;
}

std::unique_ptr<mlir::ExecutionEngine> compileCPU(mlir::ModuleOp module,
                                                std::string *loweredLLVMIR,
                                                bool shaped = false) {
  lowerToLLVM(module, shaped);
  module.walk([&](mlir::Operation *op) {
    if (op != module.getOperation() &&
        op->getDialect()->getNamespace() != "llvm")
      throw std::runtime_error("CPU lowering left non-LLVM operations");
  });
  if (loweredLLVMIR) {
    llvm::raw_string_ostream stream(*loweredLLVMIR);
    module.print(stream);
  }
  mlir::ExecutionEngineOptions options;
  // ExecutionEngineOptions holds a non-owning function_ref.
  auto transformer = mlir::makeOptimizingTransformer(2, 0, nullptr);
  options.transformer = transformer;
  auto engine = mlir::ExecutionEngine::create(module, options);
  if (!engine)
    throwLLVMError(engine.takeError(), "could not create CPU JIT");
  return std::move(*engine);
}

void invokeFlatBuffers(void *function, std::span<const TensorArgument> arguments,
                       std::vector<int32_t> &output) {
  // MLIR's descriptor ABI uses mutable pointers even for read-only inputs.
  // The verified map reads input and writes only the independent output.
  using Descriptor = StridedMemRefType<int32_t, 1>;
  std::array<Descriptor, 3> inputs;
  for (size_t i = 0; i < arguments.size(); ++i) {
    auto *data = const_cast<int32_t *>(arguments[i].values.data());
    inputs[i] = {data, data, 0, {static_cast<int64_t>(arguments[i].values.size())}, {1}};
  }
  Descriptor out{output.data(), output.data(), 0,
                 {static_cast<int64_t>(output.size())}, {1}};
  // Match the generated C-interface signature exactly for each supported arity.
  switch (arguments.size()) {
  case 1:
    reinterpret_cast<void (*)(Descriptor *, Descriptor *)>(function)(&inputs[0], &out);
    break;
  case 2:
    reinterpret_cast<void (*)(Descriptor *, Descriptor *, Descriptor *)>(function)(
        &inputs[0], &inputs[1], &out);
    break;
  case 3:
    reinterpret_cast<void (*)(Descriptor *, Descriptor *, Descriptor *, Descriptor *)>(function)(
        &inputs[0], &inputs[1], &inputs[2], &out);
    break;
  }
}

} // namespace

std::vector<int32_t> runTensorLiteralCPU(const std::string &source,
                                         std::string *loweredLLVMIR) {
  auto context = makeCPUContext();
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(context.get()));
  translateTensorSource(*module, source);

  auto engine = compileCPU(*module, loweredLLVMIR);

  auto maybeFunction = engine->lookup("_mlir_ciface_tensor_literal");
  if (!maybeFunction)
    throwLLVMError(maybeFunction.takeError(), "CPU JIT lookup failed");
  using TensorFunction = void (*)(StridedMemRefType<int32_t, 1> *);
  auto functionPointer = reinterpret_cast<TensorFunction>(*maybeFunction);
  StridedMemRefType<int32_t, 1> descriptor{};
  functionPointer(&descriptor);

  std::vector<int32_t> output;
  output.reserve(descriptor.sizes[0]);
  for (int64_t i = 0; i < descriptor.sizes[0]; ++i)
    output.push_back(
        descriptor.data[descriptor.offset + i * descriptor.strides[0]]);
  std::free(descriptor.basePtr);
  return output;
}

std::vector<int32_t> runTensorCPU(const std::string &expression,
                                 std::span<const TensorArgument> arguments,
                                 std::string *loweredLLVMIR) {
  if (loweredLLVMIR)
    loweredLLVMIR->clear();
  std::vector<TensorArgumentSpec> specs;
  for (const auto &argument : arguments)
    specs.push_back({argument.name, argument.values.size()});
  auto context = makeCPUContext();
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(context.get()));
  translateTensorExpression(*module, expression, specs);

  const auto input = arguments.front().values;
  if (input.size() > std::vector<int32_t>().max_size())
    throw std::invalid_argument("tensor element count exceeds host buffer range");
  // Validate even empty expressions before returning, without creating a JIT.
  if (input.empty())
    return {};
  std::vector<int32_t> output(input.size());
  auto engine = compileCPU(*module, loweredLLVMIR);
  auto function = engine->lookup("_mlir_ciface_tensor_expression");
  if (!function)
    throwLLVMError(function.takeError(), "CPU JIT lookup failed");
  invokeFlatBuffers(*function, arguments, output);
  return output;
}

ShapedTensorResult runShapedTensorCPU(
    const std::string &expression, std::span<const ShapedTensorArgument> arguments,
    std::string *loweredLLVMIR) {
  if (loweredLLVMIR)
    loweredLLVMIR->clear();
  std::vector<ShapedTensorArgumentSpec> specs;
  for (const auto &argument : arguments)
    specs.push_back({argument.name, argument.values.size(), argument.shape});
  auto context = makeCPUContext();
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(context.get()));
  auto shape = translateShapedTensorExpression(*module, expression, specs);
  const auto count =
      static_cast<size_t>(shape.size() == 1 ? shape[0] : shape[0] * shape[1]);
  ShapedTensorResult result{{}, std::move(shape)};
  if (!count)
    return result;
  result.values.resize(count);
  auto engine = compileCPU(*module, loweredLLVMIR, true);
  auto function = engine->lookup("_mlir_ciface_shaped_tensor_expression");
  if (!function)
    throwLLVMError(function.takeError(), "shaped CPU JIT lookup failed");
  std::vector<TensorArgument> flat;
  for (const auto &argument : arguments)
    flat.push_back({argument.name, argument.values});
  invokeFlatBuffers(*function, flat, result.values);
  return result;
}

} // namespace sconeml::tensor
