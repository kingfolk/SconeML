#include "test/runner/Pipelines.h"
#include "src/conversion/ClosureConversion.h"
#include "src/conversion/LowerTensorToGPU.h"
#include "src/conversion/LowerTensorToSCF.h"
#include "src/conversion/LowerToLLVM.h"
#include "src/conversion/MetalExporter.h"
#include "src/conversion/UnwrapLet.h"
#include "src/dialect/LetAlgDialect.h"
#include "src/tensor/CPURunner.h"
#include "src/tensor/GPURunner.h"
#include "src/tensor/TensorFrontend.h"

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"

#include <stdexcept>
#include <utility>

namespace sconeml::test {
namespace {
struct PipelineInputError : std::runtime_error {
  using std::runtime_error::runtime_error;
};
void pipelineRequire(bool condition, const std::string &message) {
  if (!condition)
    throw PipelineInputError(message);
}
} // namespace
std::unique_ptr<mlir::MLIRContext> makeContext() {
  mlir::DialectRegistry registry;
  mlir::registerAllToLLVMIRTranslations(registry);
  registry.insert<sconeml::letalg::LetAlgDialect, mlir::arith::ArithDialect,
                  mlir::func::FuncDialect, mlir::memref::MemRefDialect,
                  mlir::scf::SCFDialect, mlir::cf::ControlFlowDialect,
                  mlir::LLVM::LLVMDialect, mlir::gpu::GPUDialect>();
  auto context = std::make_unique<mlir::MLIRContext>(registry);
  context->loadAllAvailableDialects();
  return context;
}

static std::string print(mlir::ModuleOp module) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  module.print(stream);
  return text;
}

static sconeml::letalg::TensorMapOp findMap(mlir::ModuleOp module) {
  sconeml::letalg::TensorMapOp map;
  unsigned count = 0;
  module.walk([&](sconeml::letalg::TensorMapOp candidate) {
    map = candidate;
    ++count;
  });
  pipelineRequire(count == 1, "pipeline requires exactly one tensor_map");
  return map;
}

Output execute(const TestCase &test, const Run &run, bool disableMetal) {
  Output output;
  auto context = makeContext();
  mlir::ScopedDiagnosticHandler handler(
      context.get(), [&](mlir::Diagnostic &diag) {
        llvm::raw_string_ostream stream(output.diagnostic);
        diag.print(stream);
        stream << '\n';
        return mlir::success();
      });
  try {
    if (run.pipeline == "tensor-cpu" || run.pipeline == "tensor-metal-legacy") {
      require(test.input == "source",
              "execution pipeline requires source input");
      // Surface frontend diagnostics through this run's handler before the
      // production runner builds its own execution context.
      output.stage = "frontend";
      mlir::OwningOpRef<mlir::ModuleOp> check =
          mlir::ModuleOp::create(mlir::UnknownLoc::get(context.get()));
      sconeml::tensor::translateTensorSource(*check, test.program);
      if (run.pipeline == "tensor-cpu") {
        output.stage = "cpu-execute";
        output.number = Number{true, sconeml::tensor::runTensorLiteralCPU(
                                         test.program, &output.ir)};
      } else {
        output.stage = "metal-execute";
        if (disableMetal || !sconeml::tensor::gpuIsAvailable()) {
          output.skipped = true;
          output.diagnostic = "Metal device unavailable or disabled";
          return output;
        }
        output.number = Number{true, sconeml::tensor::runTensorLiteralGPU(
                                         test.program, &output.text)};
      }
      output.success = true;
      return output;
    }

    mlir::OwningOpRef<mlir::ModuleOp> module;
    if (test.input == "source") {
      output.stage = "frontend";
      module = mlir::ModuleOp::create(mlir::UnknownLoc::get(context.get()));
      sconeml::tensor::translateTensorSource(*module, test.program);
    } else {
      output.stage = "parse";
      auto buffer =
          llvm::MemoryBuffer::getMemBufferCopy(test.program, test.path);
      llvm::SourceMgr manager;
      manager.AddNewSourceBuffer(std::move(buffer), llvm::SMLoc());
      module = mlir::parseSourceFile<mlir::ModuleOp>(
          manager,
          mlir::ParserConfig(context.get(), /*verifyAfterParse=*/false));
      if (!module)
        return output;
    }
    if (run.pipeline == "verify") {
      output.stage = "verify";
      output.success = mlir::succeeded(mlir::verify(*module));
      return output;
    }
    const bool legacy = run.pipeline.starts_with("legacy-");
    if (legacy) {
      require(test.input == "source", "legacy pipeline requires source");
      auto function = *module->getOps<mlir::func::FuncOp>().begin();
      function.setSymName("test_function");
      function.setFunctionType(mlir::FunctionType::get(context.get(), {}, {}));
      function->removeAttr("llvm.emit_c_interface");
    }
    if (run.pipeline == "letalg-opt" || legacy ||
        run.pipeline == "scalar-cpu") {
      output.stage = "optimize";
      mlir::PassManager passes(context.get());
      passes.addPass(sconeml::createUnwrapLetPass());
      passes.addPass(sconeml::createClosureConversionPass());
      if (!legacy && run.pipeline != "scalar-cpu") {
        passes.addPass(mlir::createCanonicalizerPass());
        passes.addPass(mlir::createCSEPass());
      }
      if (mlir::failed(passes.run(*module)))
        return output;
    }
    if (run.pipeline == "letalg" || run.pipeline == "letalg-opt" ||
        run.pipeline == "legacy-letalg") {
      output.stage = "verify";
      if (mlir::failed(mlir::verify(*module)))
        return output;
      output.ir = output.text = print(*module);
    } else if (run.pipeline == "tensor-effects") {
      output.stage = "verify";
      if (mlir::failed(mlir::verify(*module)))
        return output;
      unsigned index = 0;
      module->walk([&](sconeml::letalg::TensorMapOp map) {
        auto interface =
            mlir::cast<mlir::MemoryEffectOpInterface>(map.getOperation());
        llvm::SmallVector<mlir::MemoryEffects::EffectInstance> effects;
        interface.getEffects(effects);
        output.effects += "map " + std::to_string(index++) + ":\n";
        for (auto &effect : effects) {
          std::string kind =
              mlir::isa<mlir::MemoryEffects::Read>(effect.getEffect()) ? "read"
              : mlir::isa<mlir::MemoryEffects::Write>(effect.getEffect())
                  ? "write"
                  : "other";
          bool bound = false;
          for (unsigned operand = 0; operand < map.getNumOperands(); ++operand)
            if (effect.getValue() == map->getOperand(operand)) {
              output.effects +=
                  "  " + kind + " operand " + std::to_string(operand) + "\n";
              bound = true;
            }
          if (!bound)
            output.effects += "  " + kind + " unbound\n";
        }
      });
      pipelineRequire(index != 0, "effects pipeline found no tensor maps");
    } else if (run.pipeline == "tensor-scf") {
      output.stage = "cpu-lowering";
      mlir::PassManager passes(context.get());
      passes.enableVerifier(false);
      passes.addPass(sconeml::createLowerTensorToSCFPass());
      if (mlir::failed(passes.run(*module)) ||
          mlir::failed(mlir::verify(*module)))
        return output;
      output.ir = print(*module);
    } else if (run.pipeline == "tensor-gpu-ir" ||
               run.pipeline == "tensor-msl" ||
               run.pipeline == "metal-compile") {
      auto before = print(*module);
      bool exportPreserved = true;
      mlir::OwningOpRef<mlir::ModuleOp> device;
      bool hasMap = false;
      module->walk([&](sconeml::letalg::TensorMapOp) { hasMap = true; });
      if (hasMap || run.pipeline == "tensor-gpu-ir") {
        output.stage = "gpu-lowering";
        auto result = sconeml::lowerTensorMapToGPU(findMap(*module));
        if (mlir::failed(result))
          return output;
        device = std::move(*result);
      } else {
        device = module->clone();
      }
      output.ir = print(*device);
      if (run.pipeline != "tensor-gpu-ir") {
        output.stage = "msl-export";
        auto beforeExport = print(*device);
        auto result = sconeml::exportMetalKernel(*device);
        if (mlir::failed(result))
          return output;
        exportPreserved = beforeExport == print(*device);
        output.text = result->source;
        output.entrypoint = result->entrypoint;
        if (run.pipeline == "metal-compile") {
          output.stage = "metal-compile";
          if (disableMetal || !sconeml::tensor::gpuIsAvailable()) {
            output.skipped = true;
            output.diagnostic = "Metal device unavailable or disabled";
            return output;
          }
          compileMetal(result->source, result->entrypoint);
        }
      }
      output.sourceUnchanged = exportPreserved && before == print(*module);
    } else if (run.pipeline == "scalar-cpu" || run.pipeline == "legacy-cpu") {
      output.stage = "cpu-lowering";
      auto function = *module->getOps<mlir::func::FuncOp>().begin();
      auto name = function.getSymName().str();
      function.setFunctionType(mlir::FunctionType::get(
          context.get(), {}, {mlir::IntegerType::get(context.get(), 32)}));
      mlir::PassManager passes(context.get());
      passes.addPass(sconeml::createLowerToLLVMPass());
      passes.addPass(mlir::createSCFToControlFlowPass());
      passes.addPass(mlir::createConvertControlFlowToLLVMPass());
      passes.addPass(mlir::createArithToLLVMConversionPass());
      passes.addPass(mlir::createConvertFuncToLLVMPass());
      if (mlir::failed(passes.run(*module)))
        return output;
      output.stage = "cpu-execute";
      auto engine = mlir::ExecutionEngine::create(*module);
      if (!engine)
        throw std::runtime_error(llvm::toString(engine.takeError()));
      auto functionPointer = (*engine)->lookup(name);
      if (!functionPointer)
        throw std::runtime_error(llvm::toString(functionPointer.takeError()));
      output.number =
          Number{false, {reinterpret_cast<int32_t (*)()>(*functionPointer)()}};
    } else {
      throw PipelineInputError("unimplemented pipeline adapter: " +
                               run.pipeline);
    }
    output.success = true;
  } catch (const PipelineInputError &error) {
    output.stage = "runner";
    output.diagnostic += error.what();
  } catch (const std::exception &error) {
    output.diagnostic += error.what();
  }
  return output;
}
} // namespace sconeml::test
