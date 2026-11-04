#include "src/tensor/MetalRuntime.h"

#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}
template <typename F> void rejects(const char *message, F action) {
  try {
    action();
  } catch (const std::runtime_error &error) {
    require(std::string(error.what()).find(message) != std::string::npos,
            "runtime rejected for the wrong reason");
    return;
  }
  throw std::runtime_error(std::string("runtime accepted: ") + message);
}
} // namespace

// API contracts involving synthetic capacities and u32 boundaries cannot be
// source programs: testing them must not allocate multi-gigabyte host tensors.
// Numeric GPU behavior belongs in the unified data fixtures instead.
int main() {
  using namespace sconeml::tensor;
  try {
    auto launch = planMetalLaunch(5, 5, 5, {4}, 64, 1024);
    require(launch.count == 5 && launch.byteCount == 20 &&
                launch.threadsPerThreadgroup == 4 && launch.threadgroupCount == 2,
            "incorrect padded launch geometry");
    require(planMetalLaunch(4, 4, 4, {4}).threadgroupCount == 1,
            "exact group must not gain an extra group");
    require(planMetalLaunch(1, 1, 1, {}, 32).threadsPerThreadgroup == 32,
            "automatic width must respect device limit");
    require(planMetalLaunch(1, 1, 1, {}, 1024).threadsPerThreadgroup == 256,
            "automatic width must be capped at 256");
    constexpr std::size_t u32Max = std::numeric_limits<uint32_t>::max();
    launch = planMetalLaunch(u32Max, u32Max, u32Max, {256});
    require(static_cast<uint64_t>(launch.threadgroupCount) * 256 ==
                static_cast<uint64_t>(u32Max) + 1,
            "final valid u32 thread index must be accepted");
    rejects("padded thread index", [&] {
      planMetalLaunch(u32Max, u32Max, u32Max, {7});
    });
    rejects("element count", [&] {
      planMetalLaunch(static_cast<std::size_t>(u32Max) + 1, u32Max, u32Max);
    });
    rejects("buffer capacity", [] { planMetalLaunch(5, 4, 5); });
    rejects("buffer capacity", [] { planMetalLaunch(5, 5, 4); });
    rejects("byte size exceeds device limit", [] {
      planMetalLaunch(5, 5, 5, {4}, 64, 19);
    });
    rejects("threads per threadgroup", [] {
      planMetalLaunch(5, 5, 5, {65}, 64);
    });
    rejects("threads per threadgroup", [] {
      planMetalLaunch(1, 1, 1, {}, 0);
    });
    rejects("threads per threadgroup", [&] {
      planMetalLaunch(1, 1, 1, {static_cast<std::size_t>(u32Max) + 1});
    });

    // Invalid source and no entrypoint are intentional: an empty runtime call
    // must return before compilation/device access, including on the stub.
    runMetalKernel("invalid Metal source", "", {}, {}, 0);
    require(planMetalLaunch(0, 0, 0).threadgroupCount == 0,
            "empty input must not dispatch");
    std::vector<int32_t> input{1}, output{123};
    rejects("buffer capacity", [&] {
      runMetalKernel("", "", input, output, 2);
    });
    require(output[0] == 123, "rejected launch modified host output");
#ifdef SCONEML_EXPECT_METAL_STUB
    require(!metalIsAvailable(), "stub unexpectedly reports an available device");
    rejects("only available on macOS", [&] {
      runMetalKernel("", "tensor_map", input, output, 1);
    });
#endif
    if (metalIsAvailable()) {
      rejects("Metal compilation failed", [&] {
        runMetalKernel("invalid Metal source", "tensor_map", input, output, 1);
      });
      // An empty library is valid MSL but contains no function. This checks
      // lookup of the caller-supplied entrypoint without another compute case.
      rejects("Metal kernel entrypoint was not generated: missing_entrypoint", [&] {
        runMetalKernel("// empty library\n", "missing_entrypoint", input, output, 1);
      });
      require(output[0] == 123, "compilation failure modified host output");
      std::cout << "PASS Metal compilation/entrypoint error propagation on "
                << metalDeviceName() << '\n';
    } else {
      std::cout << "SKIP Metal compilation/entrypoint error propagation: no device\n";
    }
    std::cout << "PASS Metal launch preflight and empty-runtime contracts\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
