#include "src/tensor/MetalRuntime.h"

#include <stdexcept>

namespace sconeml::tensor {

bool metalIsAvailable() { return false; }
std::string metalDeviceName() { return "unavailable (Metal requires macOS)"; }
void runMetalKernel(const std::string &, const std::string &,
                    std::size_t exportedInputCount,
                    std::span<const MetalInput> inputs, std::span<int32_t> output,
                    std::size_t count, MetalLaunchOptions options) {
  validateMetalInputs(exportedInputCount, inputs, count);
  planMetalLaunch(count, count, output.size(), options);
  if (count == 0)
    return;
  throw std::runtime_error("the Metal GPU backend is only available on macOS");
}

} // namespace sconeml::tensor
