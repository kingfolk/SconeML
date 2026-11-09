#ifndef SCONEML_TEST_SPEC_H
#define SCONEML_TEST_SPEC_H

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace sconeml::test {
struct InputArgument {
  std::string name;
  std::vector<int32_t> values;
  std::optional<std::vector<int64_t>> shape;
};
struct Run {
  std::string name;
  std::string pipeline;
  std::optional<uint64_t> threadsPerThreadgroup = std::nullopt;
  std::optional<std::vector<InputArgument>> arguments = std::nullopt;
};
struct Assertion {
  std::string kind;
  std::string value;
};
struct TestCase {
  std::string path;
  std::string name; // Empty for an ungrouped file.
  std::size_t programLine = 1;
  std::string id() const { return path + (name.empty() ? "" : "#" + name); }
  std::string input;
  std::string program;
  // Test-only host values for a raw rank-1 map fixture; not source parameters.
  std::optional<std::vector<int32_t>> tensorInput;
  std::vector<Run> runs;
  std::map<std::string, std::vector<Assertion>> assertions;
};
struct ExpectedError {
  std::string stage, op, message;
};
struct Number {
  bool tensor = false;
  std::vector<int32_t> values;
  bool operator==(const Number &) const = default;
};
std::vector<TestCase> readCases(const std::string &path);
std::vector<TestCase> parseCases(const std::string &path,
                                 const std::string &text);
// Single-case consumers must explicitly select a case from grouped files.
TestCase readCase(const std::string &path, const std::string &name = "");
TestCase parseCase(const std::string &path, const std::string &text,
                   const std::string &name = "");
ExpectedError parseError(const std::string &text);
std::vector<int64_t> parseShape(const std::string &text);
Number parseNumber(const std::string &text);
std::string formatNumber(const Number &number);
bool knownPipeline(const std::string &name);
std::string trim(const std::string &text);
void require(bool condition, const std::string &message);
} // namespace sconeml::test

#endif
