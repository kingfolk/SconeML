#ifndef SCONEML_CPU_RUNNER_H
#define SCONEML_CPU_RUNNER_H

#include <cstdint>
#include <string>
#include <vector>

namespace sconeml::tensor {

std::vector<int32_t> runTensorLiteralCPU(
    const std::string &source, std::string *loweredLLVMIR = nullptr);

} // namespace sconeml::tensor

#endif
