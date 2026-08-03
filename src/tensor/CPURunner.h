#ifndef SCONEML_CPU_RUNNER_H
#define SCONEML_CPU_RUNNER_H

#include <string>
#include <vector>

namespace sconeml::tensor {

std::vector<float> runPolynomialCPU(const std::vector<float> &input,
                                    std::string *loweredLLVMIR = nullptr);

} // namespace sconeml::tensor

#endif
