#include <memory>
#include <filesystem>
#include <fstream>
#include <iostream>
#include "LetAlgDialect.h"
#include "Parser.h"
#include "AstToLetAlg.h"
#include "Conversion/UnwrapLet.h"
#include "Conversion/ClosureConversion.h"

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
namespace fs = std::filesystem;

// Function to read a single file
std::string readFile(const fs::path& filepath) {
  std::ifstream file(filepath);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open file: " + filepath.string());
  }
  
  std::string content;
  std::string line;
  while (std::getline(file, line)) {
    content += line + "\n";
  }
  return content;
}

std::vector<std::pair<std::string, std::string>> readFilesWithExtensions(
  const std::string& directoryPath, 
  const std::vector<std::string>& extensions) {
  
  std::vector<std::pair<std::string, std::string>> fileContents;
  
  try {
    fs::path absPath = fs::absolute(directoryPath);
    std::cout << absPath << std::endl;
    if (!fs::exists(absPath)) {
        throw std::runtime_error("Directory does not exist or is not a directory: " + directoryPath);
    }

    for (const auto& entry : fs::directory_iterator(absPath)) {
      if (entry.is_regular_file()) {
        std::string ext = entry.path().extension().string();

        // Check if file has one of the desired extensions
        bool hasValidExtension = false;
        for (const auto& validExt : extensions) {
          if (ext == validExt) {
            hasValidExtension = true;
            break;
          }
        }
        
        if (hasValidExtension) {
          try {
            std::string filename = entry.path().filename().string();
            std::string content = readFile(entry.path());
            fileContents.emplace_back(filename, content);

            std::cout << "Read file: " << filename << " (" << content.size() << " characters)\n";
          } catch (const std::exception& e) {
            std::cerr << "Error reading file " << entry.path() << ": " << e.what() << "\n";
          }
        }
      }
    }
  } catch (const std::exception& e) {
    std::cerr << "Error accessing directory: " << e.what() << "\n";
    return {};
  }
  
  return fileContents;
}

std::string trim(const std::string& str) {
  const std::string whitespace = " \t\n\r\f\v";

  size_t start = str.find_first_not_of(whitespace);
  if (start == std::string::npos) {
    return "";
  }
  
  size_t end = str.find_last_not_of(whitespace);
  return str.substr(start, end - start + 1);
}

int main(int argc, char **argv) {
  // Register any command line options.
  registerAsmPrinterCLOptions();
  registerMLIRContextCLOptions();
  
  // Initialize LLVM.
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  std::vector<std::string> inputs {
    "let x = 1 in x + let x = 2 in x",
    "let x = 1 in x + let y = 2 in y + 10"
  };

  auto files = readFilesWithExtensions("test", {".ml"});
  for (auto file : files) {
    auto context = std::make_unique<MLIRContext>();
  
    // Load dialects including our letalg dialect
    context->getOrLoadDialect<sconeml::letalg::LetAlgDialect>();
    context->getOrLoadDialect<func::FuncDialect>();
    context->getOrLoadDialect<arith::ArithDialect>();
    context->getOrLoadDialect<scf::SCFDialect>();

    // Create a simple program using our dialect
    OpBuilder builder(context.get());
    auto loc = builder.getUnknownLoc();

    auto filename = std::get<0>(file);
    auto input = std::get<1>(file);
    int assertStart = input.find("@");
    int assertEnd = input.find("*)");
    std::string assert = input.substr(assertStart, assertEnd-assertStart);
    int expectedStart = assert.find("\n");
    std::string expected = trim(assert.substr(expectedStart+1, assert.size()));
    input = input.substr(assertEnd+2, input.size());

    std::cout << std::endl;
    std::cout << "<<<< test run for file: " << filename << ">>>>" << std::endl;
    // std::cout << "input: " << input << "assert: " << assert << std::endl;

    auto module = builder.create<ModuleOp>(loc);
    builder.setInsertionPointToStart(module.getBody());

    // Create a function that uses our letalg operations
    auto funcType = builder.getFunctionType({}, {});
    auto function = builder.create<func::FuncOp>(loc, "test_function", funcType);
    
    auto &entryBlock = *function.addEntryBlock();
    builder.setInsertionPointToStart(&entryBlock);

    auto expr = sconeml::parse(input);
    auto last = sconeml::translate(builder, expr.get());
    builder.create<sconeml::letalg::YieldOp>(loc, last.getType(), last);

    // std::cout << "LetAlg MLIR:\n";
    // module.dump();

    mlir::PassManager pm(context.get());
    // Add your custom pass to the pass manager
    pm.addPass(sconeml::createUnwrapLetPass());
    pm.addPass(sconeml::createClosureConversionPass());
    if (mlir::failed(pm.run(module))) {
      llvm::errs() << "Pass run failed\n";
      return 1;
    }

    std::string output;
    llvm::raw_string_ostream os(output);
    module.print(os);
    // std::cout << "After passes MLIR:\n";
    // std::cout << trim(output);
    if (trim(output) != expected) {
      llvm::errs() << "Assert failed for " << filename << ". Expected:\n" << expected << "\n But actual:\n" << trim(output);
      return 1;
    }

    // Verify the module
    if (failed(verify(module))) {
      llvm::errs() << "Module verification failed\n";
      return 1;
    }

    module.erase();
  }

  std::cout << "All tests passes" << std::endl;

  return 0;
}
