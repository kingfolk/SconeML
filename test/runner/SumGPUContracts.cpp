#include "src/tensor/SumGPU.h"
#include "test/runner/TestSpec.h"
#include <array>
#include <iostream>
using namespace sconeml::tensor;
using namespace sconeml::test;
namespace {
template <typename F> void rejects(const char *message, F action) {
  try {
    action();
  } catch (const std::exception &e) {
    require(std::string(e.what()).find(message) != std::string::npos, e.what());
    return;
  }
  throw std::runtime_error("accepted invalid sum GPU contract");
}
} // namespace
int main(int argc, char **argv) {
  try {
    require(argc == 3, "sum and composition fixtures required");
    auto test = readCase(argv[1], "rows");
    auto data = test.runs.front().arguments->front().values;
    auto original = data;
    std::array<ShapedTensorArgument, 1> args{{{"a", data, {2, 3}}}};
    auto plan = makeSumPlan(std::array<int64_t, 2>{2, 3}, 1);
    auto program = exportSumMetal(plan, false);
    validateSumMetal(program, MetalElementType::I32, 6, 2);
    auto bad = program;
    bad.width = 128;
    rejects("canonical",
            [&] { validateSumMetal(bad, MetalElementType::I32, 6, 2); });
    bad = program;
    bad.elementBytes = 8;
    rejects("canonical",
            [&] { validateSumMetal(bad, MetalElementType::I32, 6, 2); });
    bad = program;
    bad.source += " ";
    rejects("canonical",
            [&] { validateSumMetal(bad, MetalElementType::I32, 6, 2); });
    bad = program;
    bad.dispatches[0].inputStride = 3;
    rejects("canonical",
            [&] { validateSumMetal(bad, MetalElementType::I32, 6, 2); });
    bad = program;
    bad.profile = "f32";
    rejects("canonical",
            [&] { validateSumMetal(bad, MetalElementType::I32, 6, 2); });
    rejects("dtype",
            [&] { validateSumMetal(program, MetalElementType::F32, 6, 2); });
    rejects("lengths",
            [&] { validateSumMetal(program, MetalElementType::I32, 5, 2); });
    rejects("byte size", [&] {
      validateSumMetal(program, MetalElementType::I32, 6, 2, 23);
    });
    rejects("width 256", [&] { exportSumMetal(plan, false, {4}); });
    auto large =
        exportSumMetal(makeSumPlan(std::array<int64_t, 1>{65537}, {}), false);
    require(large.dispatches.size() == 3 && large.dispatches[0].parts == 257 &&
                large.dispatches[1].parts == 2 &&
                large.dispatches[2].parts == 1,
            "multi-dispatch dependency plan");
    auto broken = large;
    broken.dispatches.pop_back();
    rejects("canonical",
            [&] { validateSumMetal(broken, MetalElementType::I32, 65537, 1); });
    broken = large;
    std::swap(broken.dispatches[0], broken.dispatches[1]);
    rejects("canonical",
            [&] { validateSumMetal(broken, MetalElementType::I32, 65537, 1); });
    broken = large;
    broken.dispatches[0].groupCount += 1;
    rejects("canonical",
            [&] { validateSumMetal(broken, MetalElementType::I32, 65537, 1); });
    std::array<ShapedTensorArgument, 1> empty{{{"a", {}, {2, 0}}}};
    std::string dump = "stale";
    auto zero = runSumI32GPU("sum(a, axis=1)", empty, &dump);
    require(zero.values == std::vector<int32_t>({0, 0}) &&
                zero.shape == std::vector<int64_t>({2}) && dump.empty(),
            "empty GPU reduction identity");
#ifdef SCONEML_EXPECT_METAL_STUB
    auto composition = readCase(argv[2], "overlap-i32");
    std::vector<ShapedTensorArgument> composedArgs;
    for (const auto &a : *composition.runs.front().arguments)
      composedArgs.push_back({a.name, a.values, *a.shape});
    rejects("only available on macOS",
            [&] { runSumI32GPU(composition.program, composedArgs); });
    rejects("only available on macOS",
            [&] { runSumI32GPU(test.program, args); });
#else
    try {
      auto result = runSumI32GPU(test.program, args, &dump);
      Number expected;
      for (const auto &a : test.assertions.at("cpu"))
        if (a.kind == "result")
          expected = parseNumber(a.value);
      require(Number{true, result.values} == expected && data == original,
              "GPU sum direct ABI/result");
      data[0] = 99;
      std::string changedDump;
      auto changed = runSumI32GPU(test.program, args, &changedDump);
      require(changed.values != result.values && dump == changedDump,
              "sum shader embedded values or result alias");
      auto composition = readCase(argv[2], "overlap-i32");
      auto shared = composition.runs.front().arguments->front().values;
      const auto unchanged = shared;
      std::array<ShapedTensorArgument, 2> overlap{
          {{"a", shared, {2, 3}},
           {"b", std::span<const int32_t>(shared).subspan(1, 3), {3}}}};
      auto composed = runSumI32GPU(composition.program, overlap);
      Number composedExpected;
      for (const auto &a : composition.assertions.at("cpu"))
        if (a.kind == "result")
          composedExpected = parseNumber(a.value);
      require(Number{true, composed.values} == composedExpected &&
                  shared == unchanged,
              "materialized map/sum overlapping inputs");
      shared.clear();
      require(Number{true, composed.values} == composedExpected,
              "map/sum result lifetime");
      data.clear();
      require(Number{true, result.values} == expected,
              "GPU sum result lifetime");
    } catch (const std::exception &e) {
      if (std::string(e.what()).find("no Metal GPU is available") ==
          std::string::npos)
        throw;
      std::cout << "SKIP direct sum device execution: no Metal GPU\n";
    }
#endif
    std::cout << "Sum GPU canonical metadata, typed bytes, dispatch and "
                 "ownership contracts passed\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
