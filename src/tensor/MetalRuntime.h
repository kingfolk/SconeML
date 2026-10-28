#ifndef SCONEML_METAL_RUNTIME_H
#define SCONEML_METAL_RUNTIME_H

#include <cstddef>
#include <cstdint>
#include <string>

namespace sconeml::tensor {

bool metalIsAvailable();
std::string metalDeviceName();
void runMetalKernel(const std::string &source, const int32_t *input,
                    int32_t *output, std::size_t elementCount);

} // namespace sconeml::tensor

#endif
