#ifndef SCONEML_TEST_PIPELINES_H
#define SCONEML_TEST_PIPELINES_H
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "test/runner/TestSpec.h"
#include <memory>

namespace sconeml::test {
struct Output {
  bool success = false, skipped = false;
  std::string stage, diagnostic, ir, text, effects, entrypoint;
  std::optional<Number> number;
  std::optional<bool> sourceUnchanged;
};
std::unique_ptr<mlir::MLIRContext> makeContext();
Output execute(const TestCase &test, const Run &run, bool disableMetal);
void compileMetal(const std::string &source, const std::string &entrypoint);
// Test-only execution of a single map with explicit host data. Uses production
// lowering/export/runtime and does not extend the source language or public API.
void executeMapIR(mlir::ModuleOp module, const TestCase &test, const Run &run,
                  Output &output, bool disableMetal);
void checkAssertions(const TestCase &test, const Run &run, const Output &output,
                     const std::map<std::string, Output> &previous);
} // namespace sconeml::test
#endif
