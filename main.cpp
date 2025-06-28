#include "ExampleDialect.h"

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
  // Register any command line options.
  registerAsmPrinterCLOptions();
  registerMLIRContextCLOptions();
  
  // Initialize LLVM.
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  MLIRContext context;
  
  // Load dialects including our example dialect
  context.getOrLoadDialect<example::ExampleDialect>();
  context.getOrLoadDialect<func::FuncDialect>();
  context.getOrLoadDialect<arith::ArithDialect>();

  // Create a simple program using our dialect
  OpBuilder builder(&context);
  auto loc = builder.getUnknownLoc();
  
  auto module = builder.create<ModuleOp>(loc);
  builder.setInsertionPointToStart(module.getBody());

  // Create a function that uses our example operations
  auto funcType = builder.getFunctionType({}, {});
  auto function = builder.create<func::FuncOp>(loc, "test_function", funcType);
  
  auto &entryBlock = *function.addEntryBlock();
  builder.setInsertionPointToStart(&entryBlock);

  // Create some operations from our dialect
  auto constantOp = builder.create<example::ConstantOp>(loc, 42.0);
  auto constantOp2 = builder.create<example::ConstantOp>(loc, 24.0);
  
  auto addOp = builder.create<example::AddOp>(loc, 
                                              builder.getF64Type(),
                                              constantOp.getResult(), 
                                              constantOp2.getResult());
  
  builder.create<example::PrintOp>(loc, addOp.getResult());
  
  // Add return
  builder.create<func::ReturnOp>(loc);

  // Verify the module
  if (failed(verify(module))) {
    llvm::errs() << "Module verification failed\n";
    return 1;
  }

  // Print the generated MLIR
  llvm::outs() << "Generated MLIR:\n";
  module.print(llvm::outs());
  llvm::outs() << "\n";

  // Example MLIR code string for parsing
  const char *exampleMLIR = R"mlir(
    module {
      func.func @example() {
        %0 = example.constant 5.5 : f64
        %1 = example.constant 2.5 : f64
        %2 = example.add %0, %1 : f64
        example.print %2 : f64
        return
      }
    }
  )mlir";

  // Parse the example MLIR
  auto parseModule = parseSourceString<ModuleOp>(exampleMLIR, &context);
  if (!parseModule) {
    llvm::errs() << "Failed to parse example MLIR\n";
    return 1;
  }

  // Verify the parsed module
  if (failed(verify(*parseModule))) {
    llvm::errs() << "Parsed module verification failed\n";
    return 1;
  }

  llvm::outs() << "\nParsed MLIR:\n";
  parseModule->print(llvm::outs());
  llvm::outs() << "\n";

  llvm::outs() << "Example dialect operations executed successfully!\n";
  return 0;
}