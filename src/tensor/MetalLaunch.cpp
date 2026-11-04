#include "src/tensor/MetalRuntime.h"

#include <algorithm>
#include <stdexcept>

namespace sconeml::tensor {
MetalLaunch planMetalLaunch(std::size_t count, std::size_t inputCapacity,
                           std::size_t outputCapacity, MetalLaunchOptions options,
                           std::size_t maxThreads, std::size_t maxBufferBytes) {
  constexpr uint64_t u32Max = std::numeric_limits<uint32_t>::max();
  if (count > u32Max)
    throw std::runtime_error("Metal element count exceeds u32 range");
  if (count > inputCapacity || count > outputCapacity)
    throw std::runtime_error("Metal buffer capacity is smaller than element count");
  if (count > std::numeric_limits<std::size_t>::max() / sizeof(int32_t))
    throw std::runtime_error("Metal buffer byte size overflows size_t");
  const auto bytes = count * sizeof(int32_t);
  if (bytes > maxBufferBytes)
    throw std::runtime_error("Metal buffer byte size exceeds device limit");
  const auto width = options.threadsPerThreadgroup
                         ? options.threadsPerThreadgroup
                         : std::min<std::size_t>(256, maxThreads);
  if (!width || width > u32Max || width > maxThreads)
    throw std::runtime_error("Metal threads per threadgroup exceeds supported range");
  const auto groups = count / width + (count % width != 0);
  // A total of 2^32 padded threads is valid: the final index is UINT32_MAX.
  // Do this arithmetic in u64; both count and width have been bounded above.
  if (static_cast<uint64_t>(groups) * width > u32Max + 1)
    throw std::runtime_error("Metal padded thread index exceeds u32 range");
  return {static_cast<uint32_t>(count), bytes, width, groups};
}
} // namespace sconeml::tensor
