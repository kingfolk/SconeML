#include "src/tensor/SumGPU.h"
#include <cstring>
#include <stdexcept>
namespace sconeml::tensor {
void detail::runSumMetalBytes(const SumMetalProgram &p, MetalElementType type,
                              std::span<const std::byte> in,
                              std::span<std::byte> out) {
  if (in.size() % 4 || out.size() % 4)
    throw std::invalid_argument("sum buffer bytes are not whole elements");
  validateSumMetal(p, type, in.size() / 4, out.size() / 4);
  if ((!in.empty() && !in.data()) || (!out.empty() && !out.data()))
    throw std::invalid_argument(
        "sum nonempty buffers require nonnull pointers");
  if (p.dispatches.empty()) {
    if (!out.empty())
      std::memset(out.data(), 0, out.size());
    return;
  }
  throw std::runtime_error("sum Metal backend is only available on macOS");
}
} // namespace sconeml::tensor
