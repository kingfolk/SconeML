#include "src/tensor/MetalRuntime.h"

#include <stdexcept>

namespace sconeml::tensor {

bool metalIsAvailable() { return false; }
std::string metalDeviceName() { return "unavailable (Metal requires macOS)"; }
void runShapedMetalKernel(const std::string &, const std::string &,
                    const MetalKernelLayout &layout,
                    std::span<const MetalInput> inputs, std::span<int32_t> output,
                    std::size_t count, MetalLaunchOptions options) {
  validateShapedMetalInputs(layout, inputs, output, count, options);
  if (count == 0)
    return;
  throw std::runtime_error("the Metal GPU backend is only available on macOS");
}

} // namespace sconeml::tensor
