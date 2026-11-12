#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Parser/Parser.h"
#include "test/runner/Pipelines.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/FileCheck/FileCheck.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

namespace sconeml::test {
static std::string numericDifference(const Number &expected, const Number &actual) {
  if (expected.f32 != actual.f32)
    return "; dtype differs";
  if (expected.f32) {
    if (expected.f32Bits.size() != actual.f32Bits.size())
      return "; f32 element count differs";
    for (size_t i = 0; i < expected.f32Bits.size(); ++i) {
      Number a{true, {}, true, {expected.f32Bits[i]}},
          b{true, {}, true, {actual.f32Bits[i]}};
      if (!(a == b))
        return "; first differing f32 element [" + std::to_string(i) +
               "]: expected " + formatNumber(a) + ", actual " + formatNumber(b);
    }
    return "";
  }
  if (expected.tensor != actual.tensor)
    return "; scalar/tensor kind differs";
  if (expected.values.size() != actual.values.size())
    return "; element count differs: expected " +
           std::to_string(expected.values.size()) + ", actual " +
           std::to_string(actual.values.size());
  for (size_t i = 0; i < expected.values.size(); ++i)
    if (expected.values[i] != actual.values[i])
      return "; first differing element [" + std::to_string(i) + "]: expected " +
             std::to_string(expected.values[i]) + ", actual " +
             std::to_string(actual.values[i]);
  return "";
}

static void fileCheck(const std::string &expected, const std::string &actual,
                      const std::string &label) {
  llvm::FileCheckRequest request;
  request.CheckPrefixes.push_back("CHECK");
  llvm::FileCheck check(request);
  llvm::SourceMgr manager;
  std::string diagnostic;
  manager.setDiagHandler(
      [](const llvm::SMDiagnostic &diag, void *context) {
        llvm::raw_string_ostream stream(*static_cast<std::string *>(context));
        diag.print("assert", stream);
      },
      &diagnostic);
  llvm::SmallString<1024> checkText, inputText;
  auto checkBuffer = llvm::MemoryBuffer::getMemBufferCopy(expected);
  auto inputBuffer = llvm::MemoryBuffer::getMemBufferCopy(actual);
  unsigned checkId = manager.AddNewSourceBuffer(
      llvm::MemoryBuffer::getMemBufferCopy(
          check.CanonicalizeFile(*checkBuffer, checkText), label + ":expected"),
      llvm::SMLoc());
  unsigned inputId = manager.AddNewSourceBuffer(
      llvm::MemoryBuffer::getMemBufferCopy(
          check.CanonicalizeFile(*inputBuffer, inputText), label + ":actual"),
      llvm::SMLoc());
  auto pattern = manager.getMemoryBuffer(checkId)->getBuffer();
  auto input = manager.getMemoryBuffer(inputId)->getBuffer();
  require(!check.readCheckFile(manager, pattern),
          "invalid FileCheck assertion: " + diagnostic);
  require(check.checkInput(manager, input),
          diagnostic + "\nActual:\n" + actual);
}

static bool equivalent(const std::string &expected, const std::string &actual) {
  auto context = makeContext();
  auto lhs = mlir::parseSourceString<mlir::ModuleOp>(expected, context.get());
  auto rhs = mlir::parseSourceString<mlir::ModuleOp>(actual, context.get());
  require(lhs && rhs,
          "ir-equal requires valid expected and actual MLIR modules");
  mlir::IRMapping mapping;
  return mlir::OperationEquivalence::isEquivalentTo(
      *lhs, *rhs,
      [&](mlir::Value a, mlir::Value b) {
        return mlir::success(mapping.lookupOrNull(a) == b);
      },
      [&](mlir::Value a, mlir::Value b) { mapping.map(a, b); },
      mlir::OperationEquivalence::IgnoreLocations);
}

void checkAssertions(const TestCase &test, const Run &run, const Output &output,
                     const std::map<std::string, Output> &previous) {
  for (const auto &assertion : test.assertions.at(run.name)) {
    if (assertion.kind == "error") {
      auto error = parseError(assertion.value);
      require(!output.success && output.stage == error.stage &&
                  output.diagnostic.find(error.message) != std::string::npos &&
                  (error.op.empty() ||
                   output.diagnostic.find(error.op) != std::string::npos),
              "expected error at " + error.stage + " (" + error.op +
                  "): " + error.message + "\nActual stage: " + output.stage +
                  "\n" + output.diagnostic);
      continue;
    }
    require(output.success,
            "pipeline failed at " + output.stage + ": " + output.diagnostic);
    if (assertion.kind == "success")
      continue;
    if (assertion.kind == "result") {
      auto expected = parseNumber(assertion.value,
                                  (run.pipeline.starts_with("shaped-f32-") ||
                                   run.pipeline.starts_with("sum-f32-")));
      require(output.number && *output.number == expected,
              "expected " + formatNumber(expected) + ", actual " +
                  (output.number ? formatNumber(*output.number)
                                 : "no numeric result") +
                  (output.number ? numericDifference(expected, *output.number) : ""));
    } else if (assertion.kind == "shape") {
      require(output.shape &&
                  *output.shape == parseShape(assertion.value,
                                              run.pipeline.starts_with("sum-")),
              "result shape differs from expected " + assertion.value);
    } else if (assertion.kind == "compare") {
      const auto &other = previous.at(assertion.value);
      require(other.success && other.number && output.number &&
                  *other.number == *output.number && other.shape == output.shape,
              "numeric result differs from run " + assertion.value +
                  (other.number && output.number
                       ? numericDifference(*other.number, *output.number) : ""));
    } else if (assertion.kind == "source-unchanged") {
      require(output.sourceUnchanged.value_or(false),
              "input IR changed or pipeline has no preservation contract");
    } else if (assertion.kind == "ir-equal") {
      require(equivalent(assertion.value, output.ir),
              "IR structures differ\nExpected:\n" + assertion.value +
                  "\nActual:\n" + output.ir);
    } else if (assertion.kind == "ir-check" || assertion.kind == "text-check") {
      fileCheck(assertion.value,
                assertion.kind == "ir-check" ? output.ir : output.text,
                test.path + ":" + run.name);
    } else {
      const auto &actual = assertion.kind == "effects"      ? output.effects
                           : assertion.kind == "entrypoint" ? output.entrypoint
                                                            : output.text;
      require(trim(actual) == trim(assertion.value),
              "expected:\n" + assertion.value + "\nactual:\n" + actual);
    }
  }
}
} // namespace sconeml::test
