#include "src/tensor/CPURunner.h"
#include "src/tensor/GPURunner.h"
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char **argv) {
  try {
    const bool cpuOnly = argc == 2 && std::string(argv[1]) == "--cpu-only";
    struct Case { std::string source; std::vector<int32_t> expected; };
    const std::vector<Case> cases{
      {"let ts = [1, 2, 3] in ts * ts + 2 * ts + 1", {4, 9, 16}},
      {"let xs = [0, 4, 7, 11] in 10 - xs * 3", {10, -2, -11, -23}},
      {"let xs = [50000, 1] in xs * xs", {-1794967296, 1}},
    };
    if (!cpuOnly && !sconeml::tensor::gpuIsAvailable()) {
      std::cout << "SKIP: no Metal GPU available\n";
      return 77;
    }
    for (const auto &test : cases) {
      std::string llvmIR;
      const auto cpu = sconeml::tensor::runTensorLiteralCPU(test.source, &llvmIR);
      if (cpu != test.expected || llvmIR.find("llvm.func @tensor_literal") == std::string::npos)
        throw std::runtime_error("CPU tensor assertion failed: " + test.source);
      if (!cpuOnly) {
        std::string metal;
        const auto gpu = sconeml::tensor::runTensorLiteralGPU(test.source, &metal);
        if (gpu != test.expected || gpu != cpu || metal.find("kernel void tensor_map") == std::string::npos)
          throw std::runtime_error("GPU tensor assertion failed: " + test.source);
      }
    }
    std::cout << "PASS CPU" << (cpuOnly ? "" : "/GPU") << " tensor assertions\n";
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
