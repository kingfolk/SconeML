#ifndef SCONEML_GPU_RUNNER_H
#define SCONEML_GPU_RUNNER_H

#include <string>
#include <vector>

namespace sconeml::tensor {

std::vector<float> runPolynomialGPU(const std::vector<float> &input,
                                    std::string *gpuSource = nullptr);
bool gpuIsAvailable();
std::string gpuName();

} // namespace sconeml::tensor

#endif
