#ifndef SCONEML_METAL_RUNTIME_H
#define SCONEML_METAL_RUNTIME_H

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>

namespace sconeml::tensor {

bool metalIsAvailable();
std::string metalDeviceName();
using MetalInput = std::span<const int32_t>;

struct MetalLaunchOptions {
  // Zero selects min(256, the device/pipeline limit).
  std::size_t threadsPerThreadgroup = 0;
};

struct MetalLaunch {
  uint32_t count;
  std::size_t byteCount, threadsPerThreadgroup, threadgroupCount;
};

// Pure preflight, also used by the non-Apple stub. Checks capacities, bytes,
// device limits and every padded thread's u32 index without allocating buffers.
MetalLaunch planMetalLaunch(
    std::size_t elementCount, std::size_t inputCapacity,
    std::size_t outputCapacity, MetalLaunchOptions options = {},
    std::size_t maxThreads = std::numeric_limits<uint32_t>::max(),
    std::size_t maxBufferBytes = std::numeric_limits<std::size_t>::max());

// Synchronous one-shot compilation and dispatch. Empty input is a no-op even
// without a device. Only the first elementCount elements are read/written.
void runMetalKernel(const std::string &source, const std::string &entrypoint,
                    std::span<const int32_t> input, std::span<int32_t> output,
                    std::size_t elementCount, MetalLaunchOptions options = {});

// Export metadata must agree with the actual 1–3 input buffers. Each buffer
// covers elementCount elements. Inputs bind at 0..K-1, output at K, count K+1.
void validateMetalInputs(std::size_t exportedInputCount,
                         std::span<const MetalInput> inputs, std::size_t count);
void runMetalKernel(const std::string &source, const std::string &entrypoint,
                    std::size_t exportedInputCount,
                    std::span<const MetalInput> inputs, std::span<int32_t> output,
                    std::size_t elementCount, MetalLaunchOptions options = {});

} // namespace sconeml::tensor

#endif
