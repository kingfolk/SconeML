#include "src/tensor/GPURunner.h"
#include "test/runner/TestSpec.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <limits>
#include <stdexcept>

namespace {
template <typename F> void rejects(const char *message, F action) {
  try {
    action();
  } catch (const std::exception &error) {
    sconeml::test::require(std::string(error.what()).find(message) != std::string::npos,
                         "GPU API rejected for the wrong reason");
    return;
  }
  throw std::runtime_error(std::string("GPU API accepted: ") + message);
}
} // namespace

// API ownership and actual no-device behavior need direct calls rather than
// a runner skip. Numeric programs/data/expected arrays come from regular data.
int main(int argc, char **argv) {
  using namespace sconeml::test;
  using namespace sconeml::tensor;
  try {
    require(argc == 2, "usage: gpu_input_contracts <polynomial.ml>");
    const auto test = readCase(argv[1]);
    const auto &first = test.runs.at(0).arguments->at(0);
    std::array<TensorArgument, 1> empty{{{first.name, {}}}};
    std::string shader = "stale shader";
    require(runTensorGPU(test.program, empty, &shader, {4}).empty() &&
                shader.find("kernel void tensor_map") != std::string::npos &&
                shader.find("stale shader") == std::string::npos,
            "empty GPU call must export and return without device access");
    rejects("empty or missing an operand", [&] { runTensorGPU("", empty); });
    rejects("unknown tensor expression variable", [&] { runTensorGPU("missing", empty); });
    rejects("exactly one tensor argument", [&] { runTensorGPU(test.program, {}); });
    std::array<TensorArgument, 2> tooMany{{empty[0], empty[0]}};
    rejects("exactly one tensor argument", [&] { runTensorGPU(test.program, tooMany); });
    rejects("threads per threadgroup", [&] {
      runTensorGPU(test.program, empty, nullptr,
          {static_cast<std::size_t>(std::numeric_limits<uint32_t>::max()) + 1});
    });

    std::vector<int32_t> values = first.values;
    std::array<TensorArgument, 1> arguments{{{first.name, values}}};
#ifdef SCONEML_EXPECT_METAL_STUB
    require(!gpuIsAvailable(), "stub unexpectedly reports a GPU");
    rejects("only available on macOS", [&] { runTensorGPU(test.program, arguments); });
#else
    if (gpuIsAvailable()) {
      const auto &changed = test.runs.at(1).arguments->at(0);
      const auto expected = parseNumber(test.assertions.at("first").front().value).values;
      const auto changedExpected = parseNumber(test.assertions.at("changed").front().value).values;
      auto output = runTensorGPU(test.program, arguments, &shader, {4});
      require(output == expected && values == first.values,
              "GPU API result/input contract failed");
      output.front() ^= 1;
      require(values == first.values, "GPU output aliases caller input");
      output.front() ^= 1;
      values = changed.values;
      arguments[0].values = values;
      std::string otherShader;
      require(runTensorGPU(test.program, arguments, &otherShader) == changedExpected &&
                  values == changed.values && output == expected && otherShader == shader,
              "GPU repeated calls changed inputs/results or embedded data in MSL");
      std::vector<int32_t> escaped;
      {
        std::vector<int32_t> temporary = first.values;
        std::array<TensorArgument, 1> local{{{first.name, temporary}}};
        escaped = runTensorGPU(test.program, local);
      }
      require(escaped == expected, "GPU result did not survive input/device resource destruction");
      llvm::outs() << "PASS external-input GPU ownership and execution on " << gpuName() << '\n';
    } else {
      rejects("no Metal GPU is available", [&] { runTensorGPU(test.program, arguments); });
      llvm::outs() << "SKIP GPU ownership: no device; nonempty API rejection checked\n";
    }
#endif
    llvm::outs() << "PASS GPU input validation, empty and no-fallback contracts\n";
    return 0;
  } catch (const std::exception &error) {
    llvm::errs() << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
