#ifndef SCONEML_CPU_RUNNER_H
#define SCONEML_CPU_RUNNER_H

#include <cstdint>
#include <string>
#include <vector>
#include "src/tensor/TensorArgument.h"

namespace sconeml::tensor {

std::vector<int32_t> runTensorLiteralCPU(
    const std::string &source, std::string *loweredLLVMIR = nullptr);

// One-shot JIT over one read-only host tensor. The returned vector owns its
// storage. Empty input validates/translates, then returns without JIT; its
// optional LLVM dump is empty. Unsupported syntax/arguments throw.
std::vector<int32_t> runTensorCPU(
    const std::string &expression, std::span<const TensorArgument> arguments,
    std::string *loweredLLVMIR = nullptr);

} // namespace sconeml::tensor

#endif
