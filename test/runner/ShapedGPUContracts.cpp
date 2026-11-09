#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Diagnostics.h"
#include "src/conversion/LowerShapedTensorToGPU.h"
#include "src/conversion/LowerTensorToGPU.h"
#include "src/conversion/MetalExporter.h"
#include "src/dialect/LetAlgDialect.h"
#include "src/tensor/GPURunner.h"
#include "src/tensor/TensorFrontend.h"
#include "test/runner/TestSpec.h"
#include "llvm/Support/raw_ostream.h"
#include <array>
#include <limits>

using namespace sconeml::tensor;
using sconeml::test::require;
namespace {
template <typename F> void rejects(const char *message, F action) {
  try {
    action();
  } catch (const std::exception &error) {
    require(std::string(error.what()).find(message) != std::string::npos,
            std::string("wrong rejection: ") + error.what());
    return;
  }
  throw std::runtime_error(std::string("accepted: ") + message);
}
std::vector<ShapedTensorArgument>
arguments(const sconeml::test::TestCase &test) {
  std::vector<ShapedTensorArgument> result;
  for (const auto &arg : *test.runs.front().arguments)
    result.push_back({arg.name, arg.values, *arg.shape});
  return result;
}
std::vector<int32_t> expected(const sconeml::test::TestCase &test) {
  for (const auto &assertion : test.assertions.at("cpu"))
    if (assertion.kind == "result")
      return sconeml::test::parseNumber(assertion.value).values;
  throw std::runtime_error("missing fixture expectation");
}
std::string print(mlir::ModuleOp module) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  module.print(stream);
  return text;
}
} // namespace
void checkBroadcastGPUContracts(const std::string &path);

int main(int argc, char **argv) {
  try {
    require(argc == 3, "usage: shaped_gpu_contracts <semantics.ml> <broadcast-semantics.ml>");
    checkBroadcastGPUContracts(argv[2]);
    auto test = sconeml::test::readCase(argv[1], "rank2-three");
    auto args = arguments(test);
    sconeml::MetalKernelSource owned;
    {
      mlir::MLIRContext context;
      context.getOrLoadDialect<mlir::arith::ArithDialect>();
      context.getOrLoadDialect<mlir::func::FuncDialect>();
      context.getOrLoadDialect<mlir::memref::MemRefDialect>();
      context.getOrLoadDialect<sconeml::letalg::LetAlgDialect>();
      mlir::ScopedDiagnosticHandler quiet(
          &context, [](mlir::Diagnostic &) { return mlir::success(); });
      auto build = [&] {
        mlir::OwningOpRef<mlir::ModuleOp> host =
            mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
        std::vector<ShapedTensorArgumentSpec> specs;
        for (const auto &arg : args)
          specs.push_back({arg.name, arg.values.size(), arg.shape});
        translateShapedTensorExpression(*host, test.program, specs);
        return host;
      };
      auto host = build();
      const auto before = print(*host);
      auto device = sconeml::lowerShapedTensorToGPU(*host);
      require(mlir::succeeded(device) && print(*host) == before,
              "shaped extraction modified host IR");
      const auto deviceBefore = print(**device);
      auto exported = sconeml::exportMetalKernel(**device);
      require(mlir::succeeded(exported) && print(**device) == deviceBefore &&
                  exported->shapedLayout && exported->inputCount == 3 &&
                  exported->shapedLayout->inputElementCounts ==
                      std::vector<size_t>({6, 6, 6}) &&
                  exported->shapedLayout->outputElementCount == 6,
              "shaped export metadata/preservation failed");
      owned = *exported;
      // The old validated same-index shader envelope and emitted text are
      // reused.
      mlir::OwningOpRef<mlir::ModuleOp> flat =
          mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
      std::array<TensorArgumentSpec, 3> specs{{{"a", 6}, {"b", 6}, {"c", 6}}};
      translateTensorExpression(*flat, test.program, specs);
      sconeml::letalg::TensorMapOp map;
      flat->walk([&](sconeml::letalg::TensorMapOp op) { map = op; });
      auto flatDevice = sconeml::lowerTensorMapToGPU(map);
      auto flatExport = sconeml::exportMetalKernel(**flatDevice);
      require(mlir::succeeded(flatExport) && !flatExport->shapedLayout &&
                  flatExport->source == owned.source,
              "shaped extraction changed the legacy shader envelope");
      auto corrupt = [&](auto mutation) {
        mlir::OwningOpRef<mlir::ModuleOp> bad = (**device).clone();
        mlir::OpBuilder builder(&context);
        mutation(*bad, builder);
        const auto badBefore = print(*bad);
        require(mlir::failed(sconeml::exportMetalKernel(*bad)) &&
                    print(*bad) == badBefore,
                "exporter accepted or modified malformed shaped metadata");
      };
      corrupt(
          [](auto module, auto &) { module->removeAttr("sconeml.shaped"); });
      corrupt([](auto module, auto &b) {
        module->setAttr("sconeml.shaped", b.getStringAttr("bad"));
      });
      corrupt([](auto module, auto &) {
        module.walk([](mlir::gpu::GPUFuncOp kernel) {
          kernel->removeAttr("sconeml.shaped_kernel");
        });
      });
      for (int mode = 0; mode < 9; ++mode)
        corrupt([&](auto module, auto &b) {
          auto old = module->template getAttrOfType<mlir::DictionaryAttr>(
              "sconeml.shaped");
          mlir::NamedAttrList fields(old);
          if (mode == 0)
            fields.set("shape", b.getDenseI64ArrayAttr({2, 4}));
          if (mode == 1)
            fields.set("shape", b.getDenseI64ArrayAttr({-1, 6}));
          if (mode == 2)
            fields.set("input_counts", b.getDenseI64ArrayAttr({6, 5, 6}));
          if (mode == 3)
            fields.set("input_counts", b.getDenseI64ArrayAttr({6, 6}));
          if (mode == 4)
            fields.set("output_count", b.getI64IntegerAttr(7));
          if (mode == 5)
            fields.set("shape", b.getDenseI64ArrayAttr({1, 2, 3}));
          if (mode == 6)
            fields.set("extra", b.getUnitAttr());
          if (mode == 7)
            fields.set("output_count", b.getI32IntegerAttr(6));
          if (mode == 8) {
            fields.set("shape", b.getDenseI64ArrayAttr(
                                    {0, std::numeric_limits<int64_t>::max()}));
            fields.set("input_counts", b.getDenseI64ArrayAttr({0, 0, 0}));
            fields.set("output_count", b.getI64IntegerAttr(0));
          }
          module->setAttr("sconeml.shaped", fields.getDictionary(&context));
        });
      // Target-only u32 bound without constructing a forged span or huge
      // buffer.
      mlir::OwningOpRef<mlir::ModuleOp> huge =
          mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
      const size_t count =
          static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) + 1;
      std::array<ShapedTensorArgumentSpec, 1> big{
          {{"a", count, {static_cast<int64_t>(count)}}}};
      translateShapedTensorExpression(*huge, "a", big);
      rejects("exceeds u32", [&] { sconeml::validateShapedTensorGPU(*huge); });
    }
    require(owned.shapedLayout && owned.shapedLayout->outputElementCount == 6,
            "metadata did not survive MLIR context destruction");
    std::array<ShapedTensorArgument, 1> empty{{{"a", {}, {0, 3}}}};
    std::string source = "stale";
    auto zero = runShapedTensorGPU("a + 1", empty, &source);
    require(zero.values.empty() && zero.shape == std::vector<int64_t>({0, 3}) &&
                source.empty(),
            "empty shaped call requires no device and must not emit a shader");
    rejects("empty or missing an operand",
            [&] { runShapedTensorGPU("", empty); });
    std::array<ShapedTensorArgument, 2> broadcast{
        {{"a", {}, {0, 3}}, {"b", {}, {0, 1}}}};
    source = "stale";
    const auto emptyBroadcast = runShapedTensorGPU("a + b", broadcast, &source);
    require(emptyBroadcast.values.empty() && emptyBroadcast.shape == std::vector<int64_t>({0,3}) && source.empty(),
            "empty broadcast must preserve shape and skip kernel export");
#ifdef SCONEML_EXPECT_METAL_STUB
    require(!gpuIsAvailable(), "stub unexpectedly reports a device");
    rejects("only available on macOS",
            [&] { runShapedTensorGPU(test.program, args); });
#else
    if (gpuIsAvailable()) {
      auto first = runShapedTensorGPU(test.program, args);
      require(first.values == expected(test) &&
                  first.shape == std::vector<int64_t>({2, 3}),
              "direct GPU result mismatch");
      auto changed = test;
      changed.runs.front().arguments->front().values.front() = 8;
      auto later = runShapedTensorGPU(changed.program, arguments(changed));
      require(later.values.front() == 172 && first.values == expected(test),
              "later call invalidated old output");
      auto overlap = sconeml::test::readCase(argv[1], "overlap");
      auto overlapArgs = arguments(overlap);
      std::vector<int32_t> storage{1, 2, 3, 4, 5, 6, 7, 8, 999};
      overlapArgs[0].values = std::span<const int32_t>(storage).first(6);
      overlapArgs[1].values = std::span<const int32_t>(storage).subspan(2, 6);
      overlapArgs[2].values = overlapArgs[0].values;
      auto result = runShapedTensorGPU(overlap.program, overlapArgs);
      require(result.values == expected(overlap) &&
                  storage ==
                      std::vector<int32_t>({1, 2, 3, 4, 5, 6, 7, 8, 999}),
              "GPU alias/overlap or input canary failed");
      ShapedTensorResult escaped;
      {
        auto local = test;
        escaped = runShapedTensorGPU(local.program, arguments(local));
      }
      require(escaped.values == expected(test),
              "result did not outlive input/context/device resources");
      escaped.values.front() ^= 1;
      escaped.shape.front() = 9;
      require(args[0].values.front() == 1 && args[0].shape.front() == 2,
              "result aliases input storage/shape");
      // Launch from metadata that outlived its compiler context, with an output
      // canary.
      std::vector<MetalInput> inputs;
      for (const auto &arg : args)
        inputs.push_back(arg.values);
      std::vector<int32_t> out(7, 999);
      runShapedMetalKernel(owned.source, owned.entrypoint, *owned.shapedLayout,
                           inputs, out, 6);
      require(std::vector<int32_t>(out.begin(), out.begin() + 6) ==
                      expected(test) &&
                  out.back() == 999,
              "owned metadata runtime binding/readback contract failed");
      // Existing flat callers may supply longer input/output capacities and
      // still execute exactly the requested prefix.
      std::vector<int32_t> extended{1, 2, 3, 4, 5, 6, 999};
      inputs[0] = extended;
      out.assign(7, 999);
      runMetalKernel(owned.source, owned.entrypoint, owned.inputCount, inputs,
                     out, 6);
      require(std::vector<int32_t>(out.begin(), out.begin() + 6) ==
                      expected(test) &&
                  out.back() == 999,
              "legacy prefix/capacity runtime behavior changed");
      llvm::outs() << "PASS shaped GPU execution on " << gpuName() << '\n';
    } else {
      rejects("no Metal GPU", [&] { runShapedTensorGPU(test.program, args); });
      llvm::outs()
          << "SKIP shaped GPU nonempty device contracts: no Metal device\n";
    }
#endif
    llvm::outs() << "PASS shaped GPU extraction/export metadata, empty, target "
                    "bounds and ownership contracts\n";
    return 0;
  } catch (const std::exception &error) {
    llvm::errs() << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
