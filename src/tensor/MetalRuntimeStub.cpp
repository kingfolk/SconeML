#include "src/tensor/MetalRuntime.h"

#include <stdexcept>

namespace sconeml::tensor {

bool metalIsAvailable() { return false; }
std::string metalDeviceName() { return "unavailable (Metal requires macOS)"; }
void runMetalKernel(const std::string &, const float *, float *, std::size_t) {
  throw std::runtime_error("the Metal GPU backend is only available on macOS");
}

} // namespace sconeml::tensor
