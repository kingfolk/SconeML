#include "src/tensor/MetalRuntime.h"

#include <stdexcept>

namespace sconeml::tensor {

bool metalIsAvailable() { return false; }
std::string metalDeviceName() { return "unavailable (Metal requires macOS)"; }
void detail::runShapedMetalBytes(const std::string &, const std::string &,
                                 const MetalKernelLayout &layout,
                                 std::span<const MetalByteInput> inputs,
                                 std::span<std::byte> output, std::size_t count,
                                 MetalLaunchOptions options) {
  validateShapedMetalBytes(layout, inputs, output, count, options);
  if (count == 0)
    return;
  throw std::runtime_error("the Metal GPU backend is only available on macOS");
}

} // namespace sconeml::tensor
