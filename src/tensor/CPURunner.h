#ifndef SCONEML_CPU_RUNNER_H
#define SCONEML_CPU_RUNNER_H

#include <cstdint>
#include <string>
#include <vector>
#include "src/tensor/TensorArgument.h"

namespace sconeml::tensor {

std::vector<int32_t> runTensorLiteralCPU(
    const std::string &source, std::string *loweredLLVMIR = nullptr);

// One-shot JIT over 1-3 equal-length read-only host tensors. The returned vector owns its
// storage. Empty input validates/translates, then returns without JIT; its
// optional LLVM dump is empty. Unsupported syntax/arguments throw.
std::vector<int32_t> runTensorCPU(
    const std::string &expression, std::span<const TensorArgument> arguments,
    std::string *loweredLLVMIR = nullptr);

// Rank-1/rank-2 i32 execution with trailing-axis broadcasting. Empty results still
// validate source and all shapes, then return without JIT (and with empty dump).
ShapedTensorResult runShapedTensorCPU(
    const std::string &expression, std::span<const ShapedTensorArgument> arguments,
    std::string *loweredLLVMIR = nullptr);

// f32-ftz profile: RNE binary32 + / - / *, no contraction; signed FTZ on
// inputs/constants and each arithmetic result. NaN payloads are unspecified.
ShapedTensorF32Result
runShapedTensorF32CPU(const std::string &expression,
                      std::span<const ShapedTensorF32Argument> arguments,
                      std::string *loweredLLVMIR = nullptr);

} // namespace sconeml::tensor

#endif
