#include "src/tensor/MetalRuntime.h"

#include <algorithm>
#include <stdexcept>
#include <array>

namespace sconeml::tensor {
void validateMetalInputs(std::size_t exportedInputCount,
                         std::span<const MetalInput> inputs, std::size_t count) {
  if (!exportedInputCount || exportedInputCount > 3 || inputs.size() != exportedInputCount)
    throw std::runtime_error("Metal input buffer count does not match exported arity (1-3)");
  for (auto input : inputs) {
    if (input.size() < count)
      throw std::runtime_error("Metal buffer capacity is smaller than element count");
    if (count && !input.data())
      throw std::runtime_error("Metal nonempty buffers require nonnull pointers");
  }
}

void runMetalKernel(const std::string &source, const std::string &entrypoint,
                    std::span<const int32_t> input, std::span<int32_t> output,
                    std::size_t count, MetalLaunchOptions options) {
  std::array<MetalInput, 1> inputs{input};
  runMetalKernel(source, entrypoint, 1, inputs, output, count, options);
}

void runMetalKernel(const std::string &source, const std::string &entrypoint,
                    std::size_t exportedInputCount,
                    std::span<const MetalInput> inputs, std::span<int32_t> output,
                    std::size_t count, MetalLaunchOptions options) {
  validateMetalInputs(exportedInputCount, inputs, count);
  // Preserve legacy prefix/capacity semantics while the shaped API binds exact
  // specialized input lengths.
  std::vector<MetalInput> prefixes;
  for (auto input : inputs)
    prefixes.push_back(input.first(count));
  MetalKernelLayout layout{std::vector<size_t>(exportedInputCount, count), count};
  runShapedMetalKernel(source, entrypoint, layout, prefixes, output, count, options);
}

ShapedMetalLaunch planShapedMetalLaunch(const MetalKernelLayout &layout,
                                        std::span<const size_t> inputLengths,
                                        size_t outputCapacity, size_t count,
                                        MetalLaunchOptions options,
                                        size_t maxThreads,
                                        size_t maxBufferBytes) {
  if (layout.inputElementCounts.empty() ||
      layout.inputElementCounts.size() > 3 ||
      inputLengths.size() != layout.inputElementCounts.size())
    throw std::runtime_error(
        "Metal shaped input arity does not match metadata (1-3)");
  if (count != layout.outputElementCount)
    throw std::runtime_error(
        "Metal output count does not match shaped metadata");
  ShapedMetalLaunch result;
  for (size_t i = 0; i < inputLengths.size(); ++i) {
    const auto expected = layout.inputElementCounts[i];
    if (count && !expected)
      throw std::runtime_error(
          "Metal nonempty output requires nonempty shaped inputs");
    if (inputLengths[i] != expected)
      throw std::runtime_error(
          "Metal input length does not match shaped metadata");
    result.inputByteCounts.push_back(
        planMetalLaunch(expected, expected, expected, {1},
                        std::numeric_limits<uint32_t>::max(), maxBufferBytes)
            .byteCount);
  }
  result.output = planMetalLaunch(count, count, outputCapacity, options,
                                  maxThreads, maxBufferBytes);
  return result;
}

ShapedMetalLaunch validateShapedMetalInputs(const MetalKernelLayout &layout,
                                            std::span<const MetalInput> inputs,
                                            std::span<int32_t> output,
                                            size_t count,
                                            MetalLaunchOptions options) {
  std::vector<size_t> lengths;
  for (auto input : inputs)
    lengths.push_back(input.size());
  auto launch =
      planShapedMetalLaunch(layout, lengths, output.size(), count, options);
  for (auto input : inputs)
    if (!input.empty() && !input.data())
      throw std::runtime_error(
          "Metal nonempty buffers require nonnull pointers");
  if (count && !output.data())
    throw std::runtime_error("Metal nonempty buffers require nonnull pointers");
  return launch;
}

MetalLaunch planMetalLaunch(std::size_t count, std::size_t inputCapacity,
                            std::size_t outputCapacity,
                            MetalLaunchOptions options, std::size_t maxThreads,
                            std::size_t maxBufferBytes) {
  constexpr uint64_t u32Max = std::numeric_limits<uint32_t>::max();
  if (count > u32Max)
    throw std::runtime_error("Metal element count exceeds u32 range");
  if (count > inputCapacity || count > outputCapacity)
    throw std::runtime_error(
        "Metal buffer capacity is smaller than element count");
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
