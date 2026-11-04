#ifndef SCONEML_TEST_SPEC_H
#define SCONEML_TEST_SPEC_H

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace sconeml::test {
struct Run {
  std::string name;
  std::string pipeline;
  std::optional<uint64_t> threadsPerThreadgroup = std::nullopt;
};
struct Assertion {
  std::string kind;
  std::string value;
};
struct TestCase {
  std::string path;
  std::string input;
  std::string program;
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
TestCase readCase(const std::string &path);
TestCase parseCase(const std::string &path, const std::string &text);
ExpectedError parseError(const std::string &text);
Number parseNumber(const std::string &text);
std::string formatNumber(const Number &number);
bool knownPipeline(const std::string &name);
std::string trim(const std::string &text);
void require(bool condition, const std::string &message);
} // namespace sconeml::test

#endif
