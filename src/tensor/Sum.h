#ifndef SCONEML_SUM_H
#define SCONEML_SUM_H
#include "mlir/IR/BuiltinOps.h"
#include "src/tensor/TensorArgument.h"
#include "src/tensor/TensorFrontend.h"
#include <optional>

namespace sconeml::tensor {
// Scalar output is explicit: shape {} owns exactly one element. Input ranks
// remain 1/2. Sum uses a fixed 256-lane adjacent-pair tree on both backends.
template <typename T> struct SumResult {
  std::vector<T> values;
  std::vector<int64_t> shape;
};
using SumI32Result = SumResult<int32_t>;
using SumF32Result = SumResult<float>;
struct SumExpression {
  std::string map;
  std::optional<int64_t> axis;
};
struct SumPlan {
  std::vector<int64_t> inputShape, outputShape;
  std::optional<int64_t> axis;
  size_t inputCount = 0, outputCount = 0, length = 0, baseStride = 0,
         inputStride = 1;
  bool operator==(const SumPlan &) const = default;
};
SumExpression parseSumExpression(const std::string &source);
SumPlan makeSumPlan(std::span<const int64_t> shape,
                    std::optional<int64_t> axis);
void validateSumPlan(const SumPlan &plan);
// Translation never embeds input values. Returns a canonical explicit identity
// and reduction-iterator tree module, with flat input/output C ABI buffers.
void buildSumCPU(mlir::ModuleOp module, const SumPlan &plan, bool f32);
SumPlan prepareSum(const std::string &source,
                   std::span<const ShapedTensorArgumentSpec> arguments,
                   bool f32);
SumI32Result runSumI32CPU(const std::string &,
                          std::span<const ShapedTensorArgument>,
                          std::string *llvmIR = nullptr);
SumF32Result runSumF32CPU(const std::string &,
                          std::span<const ShapedTensorF32Argument>,
                          std::string *llvmIR = nullptr);
} // namespace sconeml::tensor
#endif
