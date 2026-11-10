#include "test/runner/Pipelines.h"
#include "src/conversion/ClosureConversion.h"
#include "src/conversion/LowerShapedTensorToGPU.h"
#include "src/conversion/LowerTensorToGPU.h"
#include "src/conversion/LowerTensorToSCF.h"
#include "src/conversion/LowerToLLVM.h"
#include "src/conversion/MetalExporter.h"
#include "src/conversion/UnwrapLet.h"
#include "src/dialect/LetAlgDialect.h"
#include "src/tensor/CPURunner.h"
#include "src/tensor/GPURunner.h"
#include "src/tensor/TensorFrontend.h"
#include <bit>
#include <cstring>

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
    if (run.pipeline.starts_with("shaped-f32-")) {
      output.stage = "frontend";
      std::vector<std::vector<float>> storage;
      std::vector<sconeml::tensor::ShapedTensorF32Argument> arguments;
      std::vector<sconeml::tensor::ShapedTensorArgumentSpec> specs;
      for (const auto &argument : *run.arguments)
        storage.push_back(f32Values(argument));
      for (size_t i = 0; i < storage.size(); ++i) {
        const auto &a = run.arguments->at(i);
        arguments.push_back({a.name, storage[i], *a.shape});
        specs.push_back({a.name, storage[i].size(), *a.shape});
      }
      const auto before = storage;
      mlir::OwningOpRef<mlir::ModuleOp> module =
          mlir::ModuleOp::create(mlir::UnknownLoc::get(context.get()));
      sconeml::tensor::translateShapedTensorF32Expression(*module, test.program,
                                                          specs);
      if (run.pipeline == "shaped-f32-ir")
        output.ir = print(*module);
      else {
        output.stage = "cpu-execute";
        auto result = sconeml::tensor::runShapedTensorF32CPU(
            test.program, arguments, &output.ir);
        Number number;
        number.tensor = number.f32 = true;
        for (float value : result.values)
          number.f32Bits.push_back(std::bit_cast<uint32_t>(value));
        output.number = std::move(number);
        output.shape = std::move(result.shape);
        for (size_t i = 0; i < storage.size(); ++i)
          require(std::memcmp(before[i].data(), storage[i].data(),
                              storage[i].size() * sizeof(float)) == 0,
                  "f32 execution modified input");
      }
      output.success = true;
      return output;
    }
    if (run.pipeline.starts_with("shaped-input-")) {
      output.stage = "frontend";
      std::vector<sconeml::tensor::ShapedTensorArgumentSpec> specs;
      std::vector<sconeml::tensor::ShapedTensorArgument> arguments;
      for (const auto &argument : *run.arguments) {
        specs.push_back({argument.name, argument.values.size(), *argument.shape});
        arguments.push_back({argument.name, argument.values, *argument.shape});
      }
      mlir::OwningOpRef<mlir::ModuleOp> module =
          mlir::ModuleOp::create(mlir::UnknownLoc::get(context.get()));
      sconeml::tensor::translateShapedTensorExpression(*module, test.program, specs);
      if (run.pipeline == "shaped-input-ir") {
        output.ir = print(*module);
      } else if (run.pipeline == "shaped-input-cpu") {
        output.stage = "cpu-execute";
        auto result = sconeml::tensor::runShapedTensorCPU(test.program, arguments, &output.ir);
        output.number = Number{true, std::move(result.values)};
        output.shape = std::move(result.shape);
      } else {
        output.stage = "gpu-lowering";
        auto layout = sconeml::validateShapedTensorGPU(*module);
        std::vector<size_t> lengths;
        for (const auto &argument : arguments) lengths.push_back(argument.values.size());
        sconeml::tensor::planShapedMetalLaunch(layout, lengths, layout.outputElementCount,
            layout.outputElementCount, {run.threadsPerThreadgroup.value_or(0)});
        if (run.pipeline == "shaped-input-metal") {
          output.stage = "metal-execute";
          if (layout.outputElementCount && (disableMetal || !sconeml::tensor::gpuIsAvailable())) {
            output.skipped = true;
            output.diagnostic = "Metal device unavailable or disabled";
            return output;
          }
          auto result = sconeml::tensor::runShapedTensorGPU(test.program, arguments, &output.text,
              {run.threadsPerThreadgroup.value_or(0)});
          output.number = Number{true, std::move(result.values)};
          output.shape = std::move(result.shape);
        } else {
          const auto before = print(*module);
          auto device = sconeml::lowerShapedTensorToGPU(*module);
          if (mlir::failed(device)) return output;
          output.ir = print(**device);
          output.sourceUnchanged = before == print(*module);
          if (run.pipeline == "shaped-input-msl") {
            output.stage = "msl-export";
            const auto beforeExport = print(**device);
            auto exported = sconeml::exportMetalKernel(**device);
            if (mlir::failed(exported)) return output;
            output.text = exported->source;
            output.entrypoint = exported->entrypoint;
            output.sourceUnchanged = *output.sourceUnchanged && beforeExport == print(**device);
          }
        }
      }
      output.success = true;
      return output;
    }
    if (run.pipeline == "tensor-input-ir" || run.pipeline == "tensor-input-cpu" ||
        run.pipeline == "tensor-input-metal") {
      output.stage = "frontend";
      std::vector<sconeml::tensor::TensorArgumentSpec> specs;
      std::vector<sconeml::tensor::TensorArgument> arguments;
      for (const auto &argument : *run.arguments) {
        specs.push_back({argument.name, argument.values.size()});
        arguments.push_back({argument.name, argument.values});
      }
      auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(context.get()));
      mlir::OwningOpRef<mlir::ModuleOp> owner(module);
      sconeml::tensor::translateTensorExpression(module, test.program, specs);
      if (run.pipeline == "tensor-input-ir") {
        output.ir = output.text = print(module);
      } else {
        // Exercise the public source API, not the raw-IR test adapter.
        const auto before = *run.arguments;
        if (run.pipeline == "tensor-input-metal") {
          output.stage = "metal-execute";
          const auto count = arguments.front().values.size();
          sconeml::tensor::planMetalLaunch(
              count, count, count, {run.threadsPerThreadgroup.value_or(0)});
          if (count && (disableMetal || !sconeml::tensor::gpuIsAvailable())) {
            output.skipped = true;
            output.diagnostic = "Metal device unavailable or disabled";
            return output;
          }
          output.number = Number{true, sconeml::tensor::runTensorGPU(
              test.program, arguments, &output.text,
              {run.threadsPerThreadgroup.value_or(0)})};
        } else {
          output.stage = "cpu-execute";
          output.number = Number{true, sconeml::tensor::runTensorCPU(
                                           test.program, arguments, &output.ir)};
        }
        for (size_t i = 0; i < before.size(); ++i)
          require(before[i].values == run.arguments->at(i).values,
                  "tensor runner modified a caller input");
      }
      output.success = true;
      return output;
    }
    if (run.pipeline == "tensor-cpu" || run.pipeline == "tensor-metal") {
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
                                         test.program, &output.text,
                                         {run.threadsPerThreadgroup.value_or(0)})};
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
    if (run.pipeline == "tensor-cpu-ir" || run.pipeline == "tensor-metal-ir") {
      executeMapIR(*module, test, run, output, disableMetal);
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
          auto *effectOperand = effect.getEffectValue<mlir::OpOperand *>();
          for (unsigned operand = 0; operand < map.getNumOperands(); ++operand)
            if (effectOperand ? effectOperand == &map->getOpOperand(operand)
                              : effect.getValue() == map->getOperand(operand)) {
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
