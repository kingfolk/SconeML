#include "src/tensor/Sum.h"
#include "test/runner/TestSpec.h"
#include <array>
#include <iostream>
#include <limits>
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
  throw std::runtime_error("accepted invalid sum contract");
}
} // namespace
int main(int argc, char **argv) {
  try {
    require(argc == 3, "sum and composition fixtures required");
    auto test = readCase(argv[1], "rows");
    auto data = test.runs[0].arguments->at(0).values;
    auto original = data;
    std::array<ShapedTensorArgument, 1> args{{{"a", data, {2, 3}}}};
    std::string ir, changedIR;
    auto first = runSumI32CPU(test.program, args, &ir);
    Number expected;
    for (const auto &a : test.assertions.at("cpu"))
      if (a.kind == "result")
        expected = parseNumber(a.value);
    require(Number{true, first.values} == expected && data == original,
            "direct sum ABI/input preservation");
    data[0] = 99;
    auto later = runSumI32CPU(test.program, args, &changedIR);
    require(later.values != first.values && ir == changedIR,
            "sum IR embedded values or output aliases");
    data.clear();
    args[0].shape = {99};
    require(Number{true, first.values} == expected &&
                first.shape == std::vector<int64_t>({2}),
            "sum output ownership/lifetime");
    auto zeroCase = readCase(argv[1], "empty-0");
    std::array<ShapedTensorArgument, 1> empty{{{"a", {}, {0}}}};
    ir = "stale";
    auto zero = runSumI32CPU(zeroCase.program, empty, &ir);
    require(zero.shape.empty() && zero.values == std::vector<int32_t>{0} &&
                ir.empty(),
            "rank-0 empty identity");
    auto composition = readCase(argv[2], "overlap-i32");
    auto shared = composition.runs.front().arguments->front().values;
    const auto unchanged = shared;
    std::array<ShapedTensorArgument, 2> overlap{
        {{"a", shared, {2, 3}},
         {"b", std::span<const int32_t>(shared).subspan(1, 3), {3}}}};
    auto composed = runSumI32CPU(composition.program, overlap);
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
    auto plan = makeSumPlan(std::array<int64_t, 2>{2, 3}, 1);
    auto bad = plan;
    bad.inputStride = 3;
    rejects("metadata", [&] { validateSumPlan(bad); });
    bad = plan;
    bad.outputCount = 99;
    rejects("metadata", [&] { validateSumPlan(bad); });
    rejects("u32",
            [&] { makeSumPlan(std::array<int64_t, 2>{65536, 65536}, {}); });
    rejects("outside", [&] { makeSumPlan(std::array<int64_t, 2>{2, 3}, -1); });
    require(parseShape("[]", true).empty(), "sum scalar shape parser");
    rejects("shape assertion", [&] { parseShape("[]"); });
    std::cout
        << "Sum CPU ABI, ownership, scalar/empty and plan contracts passed\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
