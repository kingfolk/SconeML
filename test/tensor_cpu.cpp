#include "src/tensor/CPURunner.h"
#include "src/tensor/TensorIR.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void assertClose(const std::vector<float> &actual,
                 const std::vector<float> &expected) {
  if (actual.size() != expected.size())
    throw std::runtime_error("CPU returned the wrong tensor size");
  for (std::size_t i = 0; i < actual.size(); ++i)
    if (std::fabs(actual[i] - expected[i]) > 1.0e-5f)
      throw std::runtime_error("CPU tensor result mismatch at element " +
                               std::to_string(i));
}
} // namespace

int main() {
  try {
    std::string ir = sconeml::tensor::buildPolynomialIR();
    if (ir.find("letalg.tensor_map") == std::string::npos ||
        ir.find("memref<?xf32>") == std::string::npos)
      throw std::runtime_error("missing tensor map IR");

    std::vector<float> input{-4.0f, -1.5f, 0.0f, 0.5f, 2.0f, 10.0f};
    std::vector<float> expected;
    for (float x : input)
      expected.push_back(x * x + 2.0f * x + 1.0f);

    std::string llvmIR;
    auto actual = sconeml::tensor::runPolynomialCPU(input, &llvmIR);
    assertClose(actual, expected);
    if (llvmIR.find("llvm.func @tensor_polynomial") == std::string::npos)
      throw std::runtime_error("CPU path did not lower to LLVM dialect");
    std::cout << "PASS CPU (MLIR -> LLVM -> native JIT)\n";
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
