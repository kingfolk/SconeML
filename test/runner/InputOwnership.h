#ifndef SCONEML_TEST_INPUT_OWNERSHIP_H
#define SCONEML_TEST_INPUT_OWNERSHIP_H

#include "src/tensor/TensorArgument.h"
#include "test/runner/TestSpec.h"

#include <array>
#include <span>
#include <stdexcept>
#include <vector>

namespace sconeml::test {
inline std::vector<int32_t> expectedInputResult(const TestCase &test,
                                               const std::string &run) {
  for (const auto &assertion : test.assertions.at(run))
    if (assertion.kind == "result")
      return parseNumber(assertion.value).values;
  throw std::runtime_error("ownership fixture requires a result assertion");
}

// Keep programs, arrays and expected values in ordinary fixtures. Only the
// address/lifetime relationships that YAML cannot express are constructed here.
template <typename Execute>
void checkThreeInputOwnership(const char *lifetimePath, const char *overlapPath,
                              Execute execute) {
  using sconeml::tensor::TensorArgument;
  const auto test = readCase(lifetimePath);
  const auto &first = *test.runs.at(0).arguments;
  const auto &changed = *test.runs.at(1).arguments;
  require(first.size() == 3 && changed.size() == 3,
          "lifetime fixture must supply three inputs");
  const auto expected = expectedInputResult(test, "cpu");
  const auto laterExpected = expectedInputResult(test, "changed");
  require(!expected.empty() && expected != laterExpected,
          "lifetime fixture must distinguish repeated calls");
  std::vector<int32_t> escaped;
  {
    constexpr int32_t canary = 0x13579bdf;
    std::array<std::vector<int32_t>, 3> guarded;
    std::array<TensorArgument, 3> arguments;
    for (size_t i = 0; i < 3; ++i) {
      guarded[i].push_back(canary);
      guarded[i].insert(guarded[i].end(), first[i].values.begin(), first[i].values.end());
      guarded[i].push_back(canary);
      arguments[i] = {first[i].name, std::span<const int32_t>(guarded[i]).subspan(
                                        1, first[i].values.size())};
    }
    const auto before = guarded;
    escaped = execute(test.program, arguments);
    require(escaped == expected && guarded == before,
            "three-input result, input immutability or input canary failed");
    escaped.front() ^= 1;
    require(guarded == before, "three-input output aliases input storage");
    escaped.front() ^= 1;
    for (size_t i = 0; i < 3; ++i) {
      require(changed[i].name == first[i].name, "changed binding order differs");
      guarded[i] = changed[i].values;
      arguments[i].values = guarded[i];
    }
    const auto laterInputs = guarded;
    const auto later = execute(test.program, arguments);
    require(later == laterExpected && escaped == expected && guarded == laterInputs,
            "three-input repeated call reused data or corrupted an earlier result");
  }
  require(escaped == expected, "three-input output did not survive input/JIT destruction");

  const auto overlap = readCase(overlapPath);
  const auto &inputs = *overlap.runs.front().arguments;
  require(inputs.size() == 3 && inputs[0].values.size() == 3,
          "overlap fixture must contain three length-three inputs");
  std::vector<int32_t> storage = inputs[0].values;
  storage.push_back(inputs[1].values.back());
  storage.push_back(inputs[2].values.back());
  std::array<TensorArgument, 3> arguments;
  for (size_t i = 0; i < 3; ++i) {
    const auto view = std::span<const int32_t>(storage).subspan(i, 3);
    require(std::vector<int32_t>(view.begin(), view.end()) == inputs[i].values,
            "overlap fixture arrays must agree with actual shared storage");
    arguments[i] = {inputs[i].name, view};
  }
  const auto before = storage;
  auto output = execute(overlap.program, arguments);
  require(output == expectedInputResult(overlap, "cpu") && storage == before,
          "partially overlapping three-input execution failed");
  output.front() ^= 1;
  require(storage == before, "overlap output aliases input storage");
}
} // namespace sconeml::test
#endif
