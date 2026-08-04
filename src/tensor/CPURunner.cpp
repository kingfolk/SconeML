#include "src/tensor/CPURunner.h"

#include "src/parser/AstToLetAlg.h"
#include "src/parser/Parser.h"
#include "src/conversion/ClosureConversion.h"
#include "src/conversion/LowerTensorToSCF.h"
#include "src/conversion/LowerToLLVM.h"
#include "src/conversion/UnwrapLet.h"

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
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

} // namespace

std::vector<int32_t> runTensorLiteralCPU(const std::string &source,
                                         std::string *loweredLLVMIR) {
  static const bool initialized = [] {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    return true;
  }();
  (void)initialized;

  mlir::DialectRegistry registry;
  mlir::registerAllToLLVMIRTranslations(registry);
  mlir::MLIRContext context(registry);
  context.getOrLoadDialect<sconeml::letalg::LetAlgDialect>();
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::cf::ControlFlowDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
  context.getOrLoadDialect<mlir::memref::MemRefDialect>();
  context.getOrLoadDialect<mlir::scf::SCFDialect>();

  mlir::OpBuilder builder(&context);
  auto loc = builder.getUnknownLoc();
  auto module = mlir::ModuleOp::create(loc);
  builder.setInsertionPointToStart(module.getBody());
  auto function = mlir::func::FuncOp::create(
      builder, loc, "tensor_literal", builder.getFunctionType({}, {}));
  function->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
  builder.setInsertionPointToStart(function.addEntryBlock());

  std::string mutableSource = source;
  auto expression = sconeml::parse(mutableSource);
  auto result = sconeml::translate(builder, expression.get());
  function.setFunctionType(builder.getFunctionType({}, {result.getType()}));
  sconeml::letalg::YieldOp::create(builder, loc, result);

  lowerToLLVM(module);
  if (loweredLLVMIR) {
    llvm::raw_string_ostream stream(*loweredLLVMIR);
    module.print(stream);
  }

  mlir::ExecutionEngineOptions options;
  auto transformer = mlir::makeOptimizingTransformer(2, 0, nullptr);
  options.transformer = transformer;
  auto maybeEngine = mlir::ExecutionEngine::create(module, options);
  if (!maybeEngine)
    throwLLVMError(maybeEngine.takeError(), "could not create CPU JIT");
  std::unique_ptr<mlir::ExecutionEngine> engine = std::move(*maybeEngine);

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

} // namespace sconeml::tensor
