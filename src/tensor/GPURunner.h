#ifndef SCONEML_GPU_RUNNER_H
#define SCONEML_GPU_RUNNER_H

#include <string>
#include <vector>
#include <cstdint>
#include "src/tensor/MetalRuntime.h"

namespace sconeml::tensor {

std::vector<int32_t> runTensorLiteralGPU(const std::string &source,
                                       std::string *gpuSource = nullptr,
                                       MetalLaunchOptions options = {});
bool gpuIsAvailable();
std::string gpuName();

} // namespace sconeml::tensor

#endif
