#include <memory>
#include "dialect/LetAlgDialect.h"
#include "parser/Parser.h"
#include "parser/AstToLetAlg.h"
#include "src/conversion/UnwrapLet.h"
#include "src/conversion/ClosureConversion.h"
#include "src/conversion/LowerToLLVM.h"

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
// #include "mlir/Transforms/Passes.h"



#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

int main(int argc, char **argv) {
  // std::string input = "let create_multiplier_adder factor offset ="
  // "let inner_function_x x ="
  //   "let innermost_function_y y ="
  //     "(x * factor) + (y + offset)"
  //   "in"
  //   "innermost_function_y"
  // "in"
  // "inner_function_x";
  // std::string input = "let x = 1 in x + let x = 2 in x";
  std::string input = "let f x = x + 10 in f 2";
  auto expr = sconeml::parse(input);

  
  // Register any command line options.
  registerAsmPrinterCLOptions();
  registerMLIRContextCLOptions();
  
  // Initialize LLVM.
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  MLIRContext context;
  
  // Load dialects including our letalg dialect
  context.getOrLoadDialect<sconeml::letalg::LetAlgDialect>();
  context.getOrLoadDialect<func::FuncDialect>();
  context.getOrLoadDialect<arith::ArithDialect>();
  context.getOrLoadDialect<scf::SCFDialect>();

  // Create a simple program using our dialect
  OpBuilder builder(&context);
  auto loc = builder.getUnknownLoc();
  
  auto module = builder.create<ModuleOp>(loc);
  builder.setInsertionPointToStart(module.getBody());

  // Create a function that uses our letalg operations
  auto funcType = builder.getFunctionType({}, {builder.getI32Type()});
  auto function = builder.create<func::FuncOp>(loc, "test_function", funcType);
  
  auto &entryBlock = *function.addEntryBlock();
  builder.setInsertionPointToStart(&entryBlock);

  auto last = sconeml::translate(builder, expr.get());
  builder.create<sconeml::letalg::YieldOp>(loc, last.getType(), last);

  llvm::outs() << "LetAlg MLIR:\n";
  module.print(llvm::outs());
  llvm::outs() << "\n";

  mlir::PassManager pm(&context);
  // Add your custom pass to the pass manager
  pm.addPass(sconeml::createUnwrapLetPass());
  pm.addPass(sconeml::createClosureConversionPass());
  if (mlir::failed(pm.run(module))) {
    llvm::errs() << "Opt Pass run failed\n";
    return 1;
  }

  llvm::outs() << "After opt passes:\n";
  module.print(llvm::outs());
  llvm::outs() << "\n";

  mlir::PassManager pm2(&context);
  pm2.addPass(sconeml::createLowerToLLVMPass());
  pm2.addPass(mlir::createConvertSCFToCFPass());
  pm2.addPass(mlir::createConvertControlFlowToLLVMPass());
  pm2.addPass(mlir::createArithToLLVMConversionPass());
  pm2.addPass(mlir::createConvertFuncToLLVMPass());
  // pm2.addPass(mlir::createReconcileUnrealizedCastsPass());

  if (mlir::failed(pm2.run(module))) {
    llvm::errs() << "LLVM Pass run failed\n";
    return 1;
  }

  llvm::outs() << "After LLVM passes:\n";
  module.print(llvm::outs());
  llvm::outs() << "\n";


  // Verify the module
  if (failed(verify(module))) {
    llvm::errs() << "Module verification failed\n";
    return 1;
  }

  return 0;
}
