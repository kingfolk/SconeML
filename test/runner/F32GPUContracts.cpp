#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "src/conversion/LowerShapedTensorToGPU.h"
#include "src/conversion/MetalExporter.h"
#include "src/tensor/GPURunner.h"
#include "src/tensor/TensorFrontend.h"
#include "test/runner/TestSpec.h"
#include <array>
#include <bit>
#include <iostream>

using namespace sconeml::tensor;
using namespace sconeml::test;
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
Number number(const std::vector<float> &v) {
  Number n{true, {}, true, {}};
  for (float x : v)
    n.f32Bits.push_back(std::bit_cast<uint32_t>(x));
  return n;
}
std::string print(mlir::ModuleOp m) {
  std::string s;
  llvm::raw_string_ostream os(s);
  m.print(os);
  return s;
}
} // namespace
int main(int argc, char **argv) {
  try {
    require(argc == 2, "f32 fixture required");
    auto fixture = readCase(argv[1], "row-broadcast");
    std::vector<std::vector<float>> storage;
    for (const auto &a : *fixture.runs.front().arguments)
      storage.push_back(f32Values(a));
    auto original = storage;
    std::vector<ShapedTensorF32Argument> args;
    std::vector<ShapedTensorArgumentSpec> specs;
    std::vector<MetalF32Input> inputs;
    for (size_t i = 0; i < storage.size(); ++i) {
      const auto &a = fixture.runs.front().arguments->at(i);
      args.push_back({a.name, storage[i], *a.shape});
      specs.push_back({a.name, storage[i].size(), *a.shape});
      inputs.push_back(storage[i]);
    }
    Number expected;
    for (const auto &a : fixture.assertions.at("cpu"))
      if (a.kind == "result")
        expected = parseNumber(a.value, true);
    sconeml::MetalKernelSource saved;
    {
      mlir::MLIRContext context;
      context.getOrLoadDialect<mlir::arith::ArithDialect>();
      context.getOrLoadDialect<mlir::func::FuncDialect>();
      context.getOrLoadDialect<mlir::memref::MemRefDialect>();
      context.getOrLoadDialect<mlir::linalg::LinalgDialect>();
      auto module = mlir::OwningOpRef<mlir::ModuleOp>(
          mlir::ModuleOp::create(mlir::UnknownLoc::get(&context)));
      translateShapedTensorF32Expression(*module, fixture.program, specs);
      auto before = print(*module);
      auto device = sconeml::lowerShapedTensorToGPU(*module);
      require(mlir::succeeded(device), "f32 lowering");
      auto deviceBefore = print(**device);
      auto kernel = sconeml::exportMetalKernel(**device);
      require(mlir::succeeded(kernel) && kernel->shapedLayout.has_value(),
              "f32 export");
      saved = *kernel;
      require(before == print(*module) && deviceBefore == print(**device),
              "f32 extraction/export mutated source");
      mlir::OpBuilder b(&context);
      auto corrupt = [&](auto mutate) {
        auto clone = mlir::OwningOpRef<mlir::ModuleOp>(
            mlir::cast<mlir::ModuleOp>((**device)->clone()));
        mutate(*clone);
        require(mlir::failed(sconeml::exportMetalKernel(*clone)),
                "accepted corrupt f32 device IR");
      };
      corrupt([&](auto m) { m->removeAttr("sconeml.float_profile"); });
      corrupt([&](auto m) {
        m->setAttr("sconeml.float_profile", b.getStringAttr("ieee"));
      });
      corrupt([&](auto m) {
        m->setAttr("sconeml.element_bytes", b.getI64IntegerAttr(8));
      });
      corrupt([&](auto m) {
        m->setAttr("sconeml.element_type", mlir::TypeAttr::get(b.getI32Type()));
      });
      corrupt([&](auto m) {
        m.walk([&](mlir::arith::MulFOp op) {
          op.setFastmath(mlir::arith::FastMathFlags::contract);
        });
      });
      corrupt([&](auto m) {
        mlir::Value first;
        m.walk([&](mlir::memref::LoadOp op) {
          if (!first)
            first = op.getResult();
        });
        m.walk([&](mlir::memref::StoreOp op) {
          op.getValueMutable().assign(first);
        });
      });
    }
    auto layout = *saved.shapedLayout;
    require(layout.elementType == MetalElementType::F32 &&
                layout.elementByteSize == sizeof(float) &&
                layout.floatProfile == MetalFloatProfile::SignedFTZ &&
                layout.inputElementCounts == std::vector<size_t>({6, 3}) &&
                layout.outputElementCount == 6,
            "owned f32 metadata");
    std::vector<float> out(6);
    auto bad = layout;
    bad.elementType = MetalElementType::I32;
    rejects("dtype", [&] {
      runShapedMetalKernelF32(saved.source, saved.entrypoint, bad, inputs, out,
                              6);
    });
    bad = layout;
    bad.elementByteSize = 8;
    rejects("byte size", [&] {
      runShapedMetalKernelF32(saved.source, saved.entrypoint, bad, inputs, out,
                              6);
    });
    bad = layout;
    bad.floatProfile = MetalFloatProfile::None;
    rejects("profile", [&] {
      runShapedMetalKernelF32(saved.source, saved.entrypoint, bad, inputs, out,
                              6);
    });
    auto shortInputs = inputs;
    shortInputs.back() = shortInputs.back().first(1);
    rejects("input length", [&] {
      runShapedMetalKernelF32(saved.source, saved.entrypoint, layout,
                              shortInputs, out, 6);
    });
    rejects("buffer capacity", [&] {
      runShapedMetalKernelF32(saved.source, saved.entrypoint, layout, inputs,
                              std::span<float>(out).first(5), 6);
    });
    rejects("output count", [&] {
      runShapedMetalKernelF32(saved.source, saved.entrypoint, layout, inputs,
                              out, 5);
    });
    std::array<size_t, 2> sizes{{6, 3}};
    rejects("byte size exceeds",
            [&] { planShapedMetalLaunch(layout, sizes, 6, 6, {4}, 64, 23); });
    std::vector<int32_t> ints(6);
    std::array<MetalInput, 2> integerInputs{
        {ints, std::span<const int32_t>(ints).first(3)}};
    rejects("dtype", [&] {
      runShapedMetalKernel(saved.source, saved.entrypoint, layout,
                           integerInputs, ints, 6);
    });
    std::array<ShapedTensorF32Argument, 2> empty{
        {{"a", {}, {0, 3}}, {"b", storage[1], {3}}}};
    std::string source = "stale";
    auto zero = runShapedTensorF32GPU(fixture.program, empty, &source, {4});
    require(zero.values.empty() && zero.shape == std::vector<int64_t>({0, 3}) &&
                source.empty(),
            "empty f32 GPU must skip device/kernel");
    rejects("expected +, - or *",
            [&] { runShapedTensorF32GPU("a / 0.5", empty); });
    rejects("threads per threadgroup", [&] {
      runShapedTensorF32GPU("a", empty, nullptr, {size_t(1) << 32});
    });
#ifdef SCONEML_EXPECT_METAL_STUB
    require(!gpuIsAvailable(), "stub reported GPU");
    rejects("only available on macOS",
            [&] { runShapedTensorF32GPU(fixture.program, args); });
#else
    if (gpuIsAvailable()) {
      runShapedMetalKernelF32(saved.source, saved.entrypoint, layout, inputs,
                              out, 6, {4});
      require(number(out) == expected,
              "typed runtime after MLIR context destruction");
      auto result = runShapedTensorF32GPU(fixture.program, args, &source, {4});
      require(number(result.values) == expected && storage == original,
              "f32 GPU ABI/value/input contract");
      storage[0][0] = 8;
      std::string changedSource;
      auto changed =
          runShapedTensorF32GPU(fixture.program, args, &changedSource, {4});
      require(changed.values.front() == 4.25f && source == changedSource &&
                  number(result.values) == expected,
              "value-dependent shader or result alias");
      storage[0] = original[0];
      auto overlapping = args;
      overlapping[1].values = std::span<const float>(storage[0]).subspan(1, 3);
      std::vector<float> row{2, 3, 4};
      auto disjoint = overlapping;
      disjoint[1].values = row;
      require(runShapedTensorF32GPU(fixture.program, overlapping).values ==
                  runShapedTensorF32GPU(fixture.program, disjoint).values,
              "overlapping f32 GPU buffers");
      storage.clear();
      args.clear();
      inputs.clear();
      require(number(result.values) == expected, "f32 result lifetime");
      std::cout << "Device: " << gpuName() << '\n';
    }
#endif
    std::cout << "f32 GPU extraction, typed metadata, runtime and ownership "
                 "contracts passed\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
