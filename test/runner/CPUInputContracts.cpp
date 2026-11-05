#include "src/tensor/CPURunner.h"
#include "src/tensor/TensorFrontend.h"
#include "test/runner/TestSpec.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <limits>
#include <stdexcept>

namespace {
template <typename F> void rejects(const char *message, F action) {
  try {
    action();
  } catch (const std::exception &error) {
    sconeml::test::require(std::string(error.what()).find(message) != std::string::npos,
                           "API rejected for the wrong reason");
    return;
  }
  throw std::runtime_error(std::string("API accepted: ") + message);
}
} // namespace

// Ownership/lifetime and an empty program cannot be expressed by the regular
// data runner. Numeric programs, inputs and expectations still come from data.
int main(int argc, char **argv) {
  using namespace sconeml::test;
  using sconeml::tensor::TensorArgument;
  using sconeml::tensor::runTensorCPU;
  try {
    require(argc == 2, "usage: cpu_input_contracts <polynomial.ml>");
    const auto test = readCase(argv[1]);
    const auto &first = test.runs.at(0).arguments->at(0);
    const auto &changed = test.runs.at(1).arguments->at(0);
    const auto expected = parseNumber(test.assertions.at("first").front().value).values;
    const auto changedExpected = parseNumber(test.assertions.at("changed").front().value).values;
    std::vector<int32_t> values = first.values;
    std::array<TensorArgument, 1> arguments{{{first.name, values}}};
    auto output = runTensorCPU(test.program, arguments);
    require(output == expected && values == first.values,
            "direct API result/input contract failed");
    output.front() ^= 1;
    require(values == first.values, "output aliases caller input");
    output.front() ^= 1;
    values = changed.values;
    arguments[0].values = values;
    const auto later = runTensorCPU(test.program, arguments);
    require(later == changedExpected && output == expected,
            "a later JIT call changed an earlier result or reused input values");

    auto carriageReturns = test.program;
    for (auto &c : carriageReturns)
      if (c == ' ')
        c = '\r';
    require(runTensorCPU(carriageReturns, arguments) == changedExpected,
            "carriage-return whitespace changed expression semantics");

    std::vector<int32_t> escaped;
    {
      std::vector<int32_t> temporary = first.values;
      std::array<TensorArgument, 1> local{{{first.name, temporary}}};
      escaped = runTensorCPU(test.program, local);
    }
    require(escaped == expected, "result did not survive input/JIT destruction");

    std::array<TensorArgument, 1> empty{{{first.name, {}}}};
    std::string dump = "old IR";
    require(runTensorCPU(test.program, empty, &dump).empty() && dump.empty(),
            "empty input must return without JIT or a stale LLVM dump");
    rejects("empty or missing an operand", [&] { runTensorCPU("", empty); });
    rejects("empty or missing an operand", [&] { runTensorCPU(" \t\r\n", empty); });
    rejects("exactly one tensor argument", [&] { runTensorCPU(test.program, {}); });
    std::array<TensorArgument, 2> tooMany{{arguments[0], arguments[0]}};
    rejects("exactly one tensor argument", [&] { runTensorCPU(test.program, tooMany); });

    // Synthetic size validation must not need an invalid span or huge allocation.
    mlir::MLIRContext context;
    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
    mlir::OwningOpRef<mlir::ModuleOp> owner(module);
    std::array<sconeml::tensor::TensorArgumentSpec, 1> huge{{
        {first.name, std::numeric_limits<std::size_t>::max()}}};
    rejects("host index/byte range", [&] {
      sconeml::tensor::translateTensorExpression(module, test.program, huge);
    });
    llvm::outs() << "PASS CPU input ownership, lifetime, empty and bounds contracts\n";
    return 0;
  } catch (const std::exception &error) {
    llvm::errs() << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
