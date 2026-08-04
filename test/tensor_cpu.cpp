#include "src/tensor/CPURunner.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main() {
  try {
    const std::string source =
        "let ts = [1, 2, 3] in ts * ts + 2 * ts + 1";
    std::string llvmIR;
    const auto actual =
        sconeml::tensor::runTensorLiteralCPU(source, &llvmIR);
    const std::vector<int32_t> expected{4, 9, 16};
    if (actual != expected)
      throw std::runtime_error("CPU tensor result mismatch");
    if (llvmIR.find("llvm.func @tensor_literal") == std::string::npos)
      throw std::runtime_error("CPU path did not lower to LLVM dialect");
    std::cout << "PASS CPU (MLIR -> LLVM -> native JIT)\n";
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
