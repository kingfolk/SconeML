#ifndef SCONEML_SHAPED_GPU_INDEXING_H
#define SCONEML_SHAPED_GPU_INDEXING_H
#include "src/tensor/MetalRuntime.h"
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>

namespace sconeml {
// The only flattened load forms supported for dense rank-1/rank-2 broadcast.
// This describes target indexing, not a general affine/view execution API.
enum class BroadcastIndex { Identity, Zero, Row, Column };
struct BroadcastIndexPlan {
  tensor::MetalKernelLayout layout;
  std::vector<BroadcastIndex> inputs;
};
inline size_t checkedGPUShapeCount(const std::vector<int64_t> &shape) {
  if (shape.empty() || shape.size() > 2)
    throw std::invalid_argument("GPU shape rank must be 1 or 2");
  for (auto extent : shape)
    if (extent < 0)
      throw std::invalid_argument("GPU shape extents must be nonnegative");
  const auto limit = std::min<uint64_t>(
      std::vector<int32_t>().max_size(),
      std::min<uint64_t>(std::numeric_limits<int64_t>::max(),
                         std::numeric_limits<size_t>::max() / sizeof(int32_t)));
  uint64_t count = 1;
  for (size_t axis = shape.size(); axis-- > 0;) {
    const auto extent = static_cast<uint64_t>(shape[axis]);
    if (extent && count > limit / extent)
      throw std::invalid_argument(
          "GPU shape exceeds host index/byte/capacity range");
    count *= extent;
  }
  if (count > std::numeric_limits<uint32_t>::max())
    throw std::invalid_argument("shaped GPU element count exceeds u32 range");
  return count;
}
inline BroadcastIndexPlan
makeBroadcastIndexPlan(const std::vector<std::vector<int64_t>> &inputs,
                       const std::vector<int64_t> &output) {
  if (inputs.empty() || inputs.size() > 3)
    throw std::invalid_argument("GPU broadcast requires 1-3 inputs");
  BroadcastIndexPlan plan;
  plan.layout.outputElementCount = checkedGPUShapeCount(output);
  size_t rank = 1;
  for (const auto &shape : inputs) {
    plan.layout.inputElementCounts.push_back(checkedGPUShapeCount(shape));
    rank = std::max(rank, shape.size());
  }
  std::vector<int64_t> inferred(rank, 1);
  for (const auto &shape : inputs)
    for (size_t axis = 0; axis < shape.size(); ++axis) {
      auto &extent = inferred[rank - shape.size() + axis];
      if (extent == shape[axis])
        continue;
      if (extent == 1)
        extent = shape[axis];
      else if (shape[axis] != 1)
        throw std::invalid_argument("incompatible GPU broadcast dimensions");
    }
  if (inferred != output)
    throw std::invalid_argument(
        "GPU output shape does not match broadcast inputs");
  for (size_t i = 0; i < inputs.size(); ++i) {
    const auto count = plan.layout.inputElementCounts[i];
    if (!plan.layout.outputElementCount)
      plan.inputs.push_back(BroadcastIndex::Zero);
    else if (count == plan.layout.outputElementCount)
      plan.inputs.push_back(BroadcastIndex::Identity);
    else if (count == 1)
      plan.inputs.push_back(BroadcastIndex::Zero);
    else if (inputs[i].size() == 2 && inputs[i].back() == 1)
      plan.inputs.push_back(BroadcastIndex::Row);
    else
      plan.inputs.push_back(BroadcastIndex::Column);
  }
  // For nonempty output C is positive and <= outputCount <= UINT32_MAX.
  // Identity < inputCount; zero < singleton count; i/C < R; i%C < C.
  // Empty output emits neither division nor remainder and never loads.
  return plan;
}
} // namespace sconeml
#endif
