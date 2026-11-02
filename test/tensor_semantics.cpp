#include "src/conversion/LowerTensorToSCF.h"
#include "src/tensor/TensorFrontend.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/raw_ostream.h"

#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const std::string &message) {
  if (!condition)
    throw std::runtime_error(message);
}

std::string mapIR(const std::string &body,
                  const std::string &inputType = "memref<4xi32>",
                  const std::string &outputType = "memref<4xi32>",
                  const std::string &arguments = "%x: i32",
                  const std::string &output = "%out") {
  return "module { func.func @test(%in: " + inputType +
         ", %out: " + outputType +
         ", %capture: i32) {\n"
         "\"letalg.tensor_map\"(%in, " +
         output + ") ({\n^bb0(" + arguments + "):\n" + body + "\n}) : (" +
         inputType + ", " + outputType + ") -> ()\nfunc.return\n} }";
}

mlir::OwningOpRef<mlir::ModuleOp> parseUnchecked(mlir::MLIRContext &context,
                                                 const std::string &source) {
  // Negative cases must reach the verifier rather than fail in the parser.
  return mlir::parseSourceString<mlir::ModuleOp>(
      source, mlir::ParserConfig(&context, /*verifyAfterParse=*/false));
}

void expectInvalid(mlir::MLIRContext &context, const std::string &source,
                   const std::string &reason) {
  auto module = parseUnchecked(context, source);
  require(bool(module), "invalid test fixture could not be parsed: " + reason);
  std::string diagnostic;
  mlir::ScopedDiagnosticHandler handler(&context, [&](mlir::Diagnostic &diag) {
    llvm::raw_string_ostream stream(diagnostic);
    diag.print(stream);
    return mlir::success();
  });
  require(mlir::failed(mlir::verify(*module)),
          "invalid map was accepted: " + reason);
  require(diagnostic.find(reason) != std::string::npos,
          "wrong diagnostic, expected '" + reason + "': " + diagnostic);
}

void testVerifier(mlir::MLIRContext &context) {
  expectInvalid(context, mapIR("%c = arith.constant 1 : i32"),
                "body must yield");
  expectInvalid(context,
                mapIR("letalg.yield %x : i32\n"
                      "^bb1(%y: i32):\nletalg.yield %y : i32"),
                "region");
  expectInvalid(
      context, mapIR("letalg.yield %x : i32", "memref<4xi32>", "memref<5xi32>"),
      "matching rank-1");
  expectInvalid(
      context,
      mapIR("letalg.yield %x : i32", "memref<2x2xi32>", "memref<2x2xi32>"),
      "matching rank-1");
  expectInvalid(context,
                mapIR("letalg.yield %x : f32", "memref<4xf32>", "memref<4xf32>",
                      "%x: f32"),
                "i32 input and output");
  expectInvalid(context,
                mapIR("letalg.yield %x : i32", "memref<4xi32>", "memref<4xi32>",
                      "%x: i32, %extra: i32"),
                "one body argument");
  expectInvalid(context,
                mapIR("letalg.yield %x : i64", "memref<4xi32>", "memref<4xi32>",
                      "%x: i64"),
                "one body argument");
  expectInvalid(context,
                mapIR("%c = arith.constant 1 : i64\n"
                      "letalg.yield %c : i64"),
                "body must yield");
  expectInvalid(context,
                mapIR("%v = arith.addi %x, %capture : i32\n"
                      "letalg.yield %v : i32"),
                "must not capture");
  expectInvalid(context, mapIR("letalg.yield %capture : i32"),
                "must not capture");
  expectInvalid(context,
                mapIR("%tmp = memref.alloca() : memref<1xi32>\n"
                      "letalg.yield %x : i32"),
                "free of memory effects");
  expectInvalid(context,
                mapIR("%v = arith.xori %x, %x : i32\n"
                      "letalg.yield %v : i32"),
                "only i32 constants and add/sub/mul");
  expectInvalid(context,
                mapIR("%v = arith.addi %x, %x overflow<nsw> : i32\n"
                      "letalg.yield %v : i32"),
                "wrapping i32");
  expectInvalid(context,
                mapIR("letalg.yield %x : i32", "memref<4xi32>", "memref<4xi32>",
                      "%x: i32", "%in"),
                "distinct input and output");
}

void testEffectsAndLowering(mlir::MLIRContext &context) {
  auto module = parseUnchecked(
      context,
      mapIR("%c = arith.constant 1 : i32\n%v = arith.addi %x, %c : i32\n"
            "letalg.yield %v : i32"));
  require(module && mlir::succeeded(mlir::verify(*module)),
          "valid map rejected");
  sconeml::letalg::TensorMapOp map;
  module->walk(
      [&](sconeml::letalg::TensorMapOp candidate) { map = candidate; });
  auto interface =
      mlir::dyn_cast<mlir::MemoryEffectOpInterface>(map.getOperation());
  require(bool(interface), "map must implement MemoryEffectOpInterface");
  llvm::SmallVector<mlir::MemoryEffects::EffectInstance> effects;
  interface.getEffects(effects);
  require(effects.size() == 2, "expected exactly input read and output write");
  bool readsInput = false, writesOutput = false;
  for (const auto &effect : effects) {
    readsInput |= mlir::isa<mlir::MemoryEffects::Read>(effect.getEffect()) &&
                  effect.getValue() == map.getInput();
    writesOutput |= mlir::isa<mlir::MemoryEffects::Write>(effect.getEffect()) &&
                    effect.getValue() == map.getOutput();
  }
  require(readsInput && writesOutput && !mlir::isMemoryEffectFree(map),
          "wrong buffer effects");

  // Both writes are observable. Neither dead-code elimination nor CSE may
  // discard/merge them merely because the maps have no SSA results.
  map->getBlock()->getOperations().insertAfter(map->getIterator(),
                                               map->clone());
  mlir::PassManager optimize(&context);
  optimize.addPass(mlir::createCanonicalizerPass());
  optimize.addPass(mlir::createCSEPass());
  require(mlir::succeeded(optimize.run(*module)), "optimization failed");
  unsigned maps = 0;
  module->walk([&](sconeml::letalg::TensorMapOp) { ++maps; });
  require(maps == 2, "optimization discarded or merged a writing map");

  mlir::PassManager lower(&context);
  lower.addPass(sconeml::createLowerTensorToSCFPass());
  require(mlir::succeeded(lower.run(*module)), "valid map lowering failed");
  maps = 0;
  unsigned loops = 0, loads = 0, stores = 0;
  module->walk([&](sconeml::letalg::TensorMapOp) { ++maps; });
  module->walk([&](mlir::scf::ForOp) { ++loops; });
  module->walk([&](mlir::memref::LoadOp) { ++loads; });
  module->walk([&](mlir::memref::StoreOp) { ++stores; });
  require(maps == 0 && loops == 2 && loads == 2 && stores == 2,
          "lowering must consume every map and preserve reads/writes");

  auto invalid = parseUnchecked(context, mapIR("%c = arith.constant 1 : i64\n"
                                               "letalg.yield %c : i64"));
  require(bool(invalid), "invalid lowering fixture could not be parsed");
  mlir::ScopedDiagnosticHandler handler(
      &context, [](mlir::Diagnostic &) { return mlir::success(); });
  mlir::PassManager reject(&context);
  reject.enableVerifier(false);
  reject.addPass(sconeml::createLowerTensorToSCFPass());
  require(
      mlir::failed(reject.run(*invalid)),
      "lowering silently accepted an invalid map with verification disabled");
}

void testStandaloneFrontend(mlir::MLIRContext &context) {
  // This executable intentionally does not link CPURunner or ExecutionEngine.
  auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
  mlir::OwningOpRef<mlir::ModuleOp> owned(module);
  sconeml::tensor::translateTensorSource(
      module, "let ts = [1, 2, 3] in ts * ts + 2 * ts + 1");
  require(mlir::succeeded(mlir::verify(module)),
          "shared frontend produced invalid IR");
  unsigned maps = 0;
  module.walk([&](sconeml::letalg::TensorMapOp) { ++maps; });
  require(maps == 1, "shared frontend must generate a tensor map");
}

} // namespace

int main() {
  try {
    mlir::DialectRegistry registry;
    registry.insert<sconeml::letalg::LetAlgDialect, mlir::arith::ArithDialect,
                    mlir::func::FuncDialect, mlir::memref::MemRefDialect,
                    mlir::scf::SCFDialect>();
    mlir::MLIRContext context(registry);
    context.loadAllAvailableDialects();
    testVerifier(context);
    testEffectsAndLowering(context);
    testStandaloneFrontend(context);
    llvm::outs() << "PASS tensor verifier, memory effects, lowering and shared "
                    "frontend\n";
    return 0;
  } catch (const std::exception &error) {
    llvm::errs() << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
