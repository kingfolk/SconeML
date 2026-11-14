#ifndef SCONEML_SUM_GPU_H
#define SCONEML_SUM_GPU_H
#include "src/tensor/MetalRuntime.h"
#include "src/tensor/Sum.h"
namespace sconeml::tensor {
struct SumDispatch {
  size_t length = 0, parts = 0, groupCount = 0, baseStride = 0, inputStride = 1;
  bool operator==(const SumDispatch &) const = default;
};
struct SumMetalProgram {
  SumPlan plan;
  MetalElementType dtype = MetalElementType::I32;
  size_t width = 256, elementBytes = 4;
  std::string profile, source;
  std::vector<SumDispatch> dispatches;
};
SumMetalProgram exportSumMetal(const SumPlan &, bool f32,
                               MetalLaunchOptions = {});
// Verifies canonical metadata, dispatch dependencies/source, typed capacities
// and checked byte limits before any device access or buffer allocation.
void validateSumMetal(
    const SumMetalProgram &, MetalElementType actualType, size_t inputElements,
    size_t outputElements,
    size_t maxBufferBytes = std::numeric_limits<size_t>::max());
namespace detail {
void runSumMetalBytes(const SumMetalProgram &, MetalElementType actualType,
                      std::span<const std::byte> input,
                      std::span<std::byte> output);
}
SumI32Result runSumI32GPU(const std::string &,
                          std::span<const ShapedTensorArgument>,
                          std::string *source = nullptr,
                          MetalLaunchOptions = {});
SumF32Result runSumF32GPU(const std::string &,
                          std::span<const ShapedTensorF32Argument>,
                          std::string *source = nullptr,
                          MetalLaunchOptions = {});
} // namespace sconeml::tensor
#endif
