#include <memory>
#include "LetAlgDialect.h"
#include "Parser.h"
#include "AstToLetAlg.h"
#include "Conversion/ClosureConversion.h"
#include "Conversion/LiftLocals.h"

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
  // std::string input = "let f x = let f1 y = y + 1 in f1 (x * 2)";
  // std::string input = "let x = 1 in let y = 2 in x + y";
  // std::string input = "let x = 1 in x + let y = 2 in y + 10";
  // std::string input = "let f x = x + 10 in f 2";
  // std::string input = "let f x y = x + y + 10 in f 2";
  // std::string input = "let x = 1 in if x then x + 10 else 0";
  // std::string input = "let a = 1 in let f x = x + 10 in f a";
  std::string input = "let a = 1 in let f x = x + a + 10 in f 2";
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
  auto funcType = builder.getFunctionType({}, {});
  auto function = builder.create<func::FuncOp>(loc, "test_function", funcType);
  
  auto &entryBlock = *function.addEntryBlock();
  builder.setInsertionPointToStart(&entryBlock);

  auto last = sconeml::translate(builder, expr.get());
  builder.create<sconeml::letalg::YieldOp>(loc, last.getType(), last);

  mlir::PassManager pm(&context);
  // Add your custom pass to the pass manager
  pm.addPass(sconeml::createClosureConversionPass());
  pm.addPass(sconeml::createLiftLocalsPass());
  if (mlir::failed(pm.run(module))) {
    llvm::errs() << "Pass run failed\n";
    return 1;
  }

  // Print the generated MLIR
  llvm::outs() << "Generated MLIR:\n";
  module.print(llvm::outs());
  llvm::outs() << "\n";

  // Verify the module
  if (failed(verify(module))) {
    llvm::errs() << "Module verification failed\n";
    return 1;
  }

  return 0;
}


// consider
// let create_multiplier_adder factor offset =
//   let inner_function_x x =
//     let innermost_function_y y =
//       (x * factor) + (y + offset)
//     in
//     innermost_function_y
//   in
//   inner_function_x
// ;;