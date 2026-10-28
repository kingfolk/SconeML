#include <memory>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include "src/dialect/LetAlgDialect.h"
#include "src/parser/Parser.h"
#include "src/parser/AstToLetAlg.h"
#include "src/conversion/UnwrapLet.h"
#include "src/conversion/ClosureConversion.h"
#include "src/conversion/LowerToLLVM.h"

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/Export.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <optional>

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

std::optional<std::string> annotationValue(const std::string &input,
                                           const std::string &marker) {
  const size_t markerStart = input.find(marker);
  if (markerStart == std::string::npos)
    return std::nullopt;
  const size_t annotationEnd = input.find("*)", markerStart);
  if (annotationEnd == std::string::npos)
    throw std::runtime_error("Unterminated annotation: " + marker);
  return trim(input.substr(markerStart + marker.size(),
                           annotationEnd - markerStart - marker.size()));
}

std::string sourceAfterAnnotations(const std::string &input) {
  size_t position = 0;
  while (true) {
    position = input.find_first_not_of(" \t\n\r", position);
    if (position == std::string::npos || input.compare(position, 2, "(*") != 0)
      return input.substr(position == std::string::npos ? input.size() : position);
    const size_t annotationEnd = input.find("*)", position + 2);
    if (annotationEnd == std::string::npos)
      throw std::runtime_error("Unterminated leading annotation block");
    position = annotationEnd + 2;
  }
}

int32_t runCPU(mlir::ModuleOp module) {
  mlir::PassManager lower(module.getContext());
  lower.addPass(sconeml::createLowerToLLVMPass());
  lower.addPass(mlir::createSCFToControlFlowPass());
  lower.addPass(mlir::createConvertControlFlowToLLVMPass());
  lower.addPass(mlir::createArithToLLVMConversionPass());
  lower.addPass(mlir::createConvertFuncToLLVMPass());
  if (mlir::failed(lower.run(module)))
    throw std::runtime_error("CPU lowering to LLVM failed");
  if (mlir::failed(mlir::verify(module)))
    throw std::runtime_error("lowered CPU module failed verification");

  auto maybeEngine = mlir::ExecutionEngine::create(module.getOperation());
  if (!maybeEngine)
    throw std::runtime_error("could not create CPU JIT: " +
                             llvm::toString(maybeEngine.takeError()));
  auto maybeFunction = (*maybeEngine)->lookup("test_function");
  if (!maybeFunction)
    throw std::runtime_error("CPU JIT lookup failed: " +
                             llvm::toString(maybeFunction.takeError()));
  using TestFunction = int32_t (*)();
  return reinterpret_cast<TestFunction>(*maybeFunction)();
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

  std::string selectedFile;
  if (argc == 3 && std::string(argv[1]) == "--file") {
    selectedFile = argv[2];
  } else if (argc != 1) {
    llvm::errs() << "usage: tester [--file <test.ml>]\n";
    return 1;
  }

  auto files = readFilesWithExtensions("test", {".ml"});
  for (auto file : files) {
    auto filename = std::get<0>(file);
    if (!selectedFile.empty() && filename != selectedFile)
      continue;

    auto input = std::get<1>(file);
    const auto letalgExpected = annotationValue(input, "@letalg:opt");
    const auto cpuExpected = annotationValue(input, "@runner:cpu");
    if (!letalgExpected && !cpuExpected) {
      llvm::errs() << "No supported annotation found in " << filename << "\n";
      return 1;
    }

    mlir::DialectRegistry registry;
    mlir::registerAllToLLVMIRTranslations(registry);
    auto context = std::make_unique<MLIRContext>(registry);
  
    // Load dialects including our letalg dialect
    context->getOrLoadDialect<sconeml::letalg::LetAlgDialect>();
    context->getOrLoadDialect<func::FuncDialect>();
    context->getOrLoadDialect<arith::ArithDialect>();
    context->getOrLoadDialect<memref::MemRefDialect>();
    context->getOrLoadDialect<cf::ControlFlowDialect>();
    context->getOrLoadDialect<LLVM::LLVMDialect>();
    context->getOrLoadDialect<scf::SCFDialect>();

    // Create a simple program using our dialect
    OpBuilder builder(context.get());
    auto loc = builder.getUnknownLoc();

    input = sourceAfterAnnotations(input);

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
    builder.create<sconeml::letalg::YieldOp>(loc, last);

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
    if (letalgExpected) {
      if (trim(output) != *letalgExpected) {
        llvm::errs() << "LetAlg assertion failed for " << filename
                     << ". Expected:\n" << *letalgExpected
                     << "\nBut actual:\n" << trim(output);
        return 1;
      }
    }

    // Verify the module
    if (failed(verify(module))) {
      llvm::errs() << "Module verification failed\n";
      return 1;
    }

    if (cpuExpected) {
      const int32_t expected = std::stoi(*cpuExpected);
      function.setType(builder.getFunctionType({}, {builder.getI32Type()}));
      const int32_t actual = runCPU(module);
      if (actual != expected) {
        llvm::errs() << "CPU runner assertion failed for " << filename
                     << ": expected " << expected << ", got " << actual << "\n";
        return 1;
      }
      std::cout << "CPU runner test passed: " << filename << "\n";
    }

    module.erase();
  }

  std::cout << "All tests passes" << std::endl;

  return 0;
}
