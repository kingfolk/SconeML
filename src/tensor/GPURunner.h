#ifndef SCONEML_GPU_RUNNER_H
#define SCONEML_GPU_RUNNER_H

#include <string>
#include <vector>
#include <cstdint>
#include "src/tensor/MetalRuntime.h"
#include "src/tensor/TensorArgument.h"

namespace sconeml::tensor {

std::vector<int32_t> runTensorLiteralGPU(const std::string &source,
                                       std::string *gpuSource = nullptr,
                                       MetalLaunchOptions options = {});
// 1-3 equal-length read-only host tensors, using the shared expression frontend.
// Returns owning host output after synchronous Metal execution. Empty inputs
// still validate/lower/export, but do not access a device or dispatch.
std::vector<int32_t> runTensorGPU(
    const std::string &expression, std::span<const TensorArgument> arguments,
    std::string *gpuSource = nullptr, MetalLaunchOptions options = {});
// Shape-specialized same-shape rank-1/rank-2 i32 Metal execution. Broadcast
// shapes are explicitly rejected until GPU broadcast indexing is implemented.
// Empty output validates but emits no kernel/MSL and needs no device.
ShapedTensorResult runShapedTensorGPU(
    const std::string &expression, std::span<const ShapedTensorArgument> arguments,
    std::string *gpuSource = nullptr, MetalLaunchOptions options = {});
bool gpuIsAvailable();
std::string gpuName();

} // namespace sconeml::tensor

#endif
