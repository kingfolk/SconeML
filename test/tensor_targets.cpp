#include "src/tensor/CPURunner.h"
#include "src/tensor/GPURunner.h"
#include "src/tensor/TensorIR.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void assertClose(const std::vector<float> &actual,
                 const std::vector<float> &expected,
                 const std::string &target) {
  if (actual.size() != expected.size())
    throw std::runtime_error(target + " returned the wrong tensor size");
  for (std::size_t index = 0; index < actual.size(); ++index) {
    if (std::fabs(actual[index] - expected[index]) > 1.0e-5f)
      throw std::runtime_error(target + " mismatch at element " +
                               std::to_string(index) + ": expected " +
                               std::to_string(expected[index]) + ", got " +
                               std::to_string(actual[index]));
  }
}

} // namespace

int main(int argc, char **argv) {
  try {
    std::string letalgIR = sconeml::tensor::buildPolynomialIR();
    if (letalgIR.find("letalg.tensor_map") == std::string::npos ||
        letalgIR.find("memref<?xf32>") == std::string::npos)
      throw std::runtime_error(
          "front end did not produce the expected local tensor LetAlg IR");

    std::vector<float> input{-4.0f, -1.5f, 0.0f, 0.5f, 2.0f, 10.0f};
    std::vector<float> expected;
    for (float value : input)
      expected.push_back(value * value + 2.0f * value + 1.0f);

    std::string llvmIR;
    std::vector<float> cpu =
        sconeml::tensor::runPolynomialCPU(input, &llvmIR);
    if (std::getenv("SCONEML_DUMP_IR"))
      std::cout << llvmIR << '\n';
    assertClose(cpu, expected, "CPU");
    if (llvmIR.find("llvm.func @tensor_polynomial") == std::string::npos)
      throw std::runtime_error("CPU path did not lower to LLVM dialect");
    std::cout << "PASS CPU (MLIR -> LLVM -> native JIT)\n";

    if (argc == 2 && std::string(argv[1]) == "--cpu-only") {
      std::cout << "All CPU tensor assertions passed\n";
      return EXIT_SUCCESS;
    }

    if (!sconeml::tensor::gpuIsAvailable())
      throw std::runtime_error("GPU assertion requested but no GPU is available");
    std::string metalSource;
    std::vector<float> gpu =
        sconeml::tensor::runPolynomialGPU(input, &metalSource);
    assertClose(gpu, expected, "GPU");
    assertClose(gpu, cpu, "CPU/GPU comparison");
    if (metalSource.find("kernel void tensor_map") == std::string::npos)
      throw std::runtime_error("GPU path did not produce a Metal kernel");
    std::cout << "PASS GPU (LetAlg tensor map -> Metal -> "
              << sconeml::tensor::gpuName() << ")\n";
    std::cout << "All cross-target tensor assertions passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
