#ifndef SCONEML_TENSOR_ARGUMENT_H
#define SCONEML_TENSOR_ARGUMENT_H

#include <cstdint>
#include <span>
#include <string>

namespace sconeml::tensor {
// Borrowed contiguous host data. Storage must remain valid until the
// synchronous runner returns; the runner never modifies or frees it.
struct TensorArgument {
  std::string name;
  std::span<const int32_t> values;
};
} // namespace sconeml::tensor

#endif
