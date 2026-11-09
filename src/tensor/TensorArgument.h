#ifndef SCONEML_TENSOR_ARGUMENT_H
#define SCONEML_TENSOR_ARGUMENT_H

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace sconeml::tensor {
// Borrowed contiguous host data. Storage must remain valid until the
// synchronous runner returns; the runner never modifies or frees it.
struct TensorArgument {
  std::string name;
  std::span<const int32_t> values;
};
// Dense row-major rank-1/rank-2 input. Shapes are copied into the result;
// values are borrowed for the duration of the synchronous call.
struct ShapedTensorArgument {
  std::string name;
  std::span<const int32_t> values;
  std::vector<int64_t> shape;
};
struct ShapedTensorResult {
  std::vector<int32_t> values;
  std::vector<int64_t> shape;
};
} // namespace sconeml::tensor

#endif
