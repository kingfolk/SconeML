#ifndef SCONEML_METAL_RUNTIME_H
#define SCONEML_METAL_RUNTIME_H

#include <cstddef>
#include <string>

namespace sconeml::tensor {

bool metalIsAvailable();
std::string metalDeviceName();
void runMetalKernel(const std::string &source, const float *input,
                    float *output, std::size_t elementCount);

} // namespace sconeml::tensor

#endif
