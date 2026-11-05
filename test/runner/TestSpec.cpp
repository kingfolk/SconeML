#include "test/runner/TestSpec.h"

#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <charconv>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>

namespace sconeml::test {
struct Config {
  int version = 0;
  std::string input;
  std::vector<Run> runs;
  std::optional<std::vector<int64_t>> tensorInput;
};
} // namespace sconeml::test

LLVM_YAML_IS_SEQUENCE_VECTOR(sconeml::test::Run)
LLVM_YAML_IS_SEQUENCE_VECTOR(sconeml::test::InputArgument)
namespace llvm::yaml {
template <> struct MappingTraits<sconeml::test::InputArgument> {
  static void mapping(IO &io, sconeml::test::InputArgument &argument) {
    io.mapRequired("name", argument.name);
    io.mapRequired("values", argument.values);
  }
};
template <> struct MappingTraits<sconeml::test::Run> {
  static void mapping(IO &io, sconeml::test::Run &run) {
    io.mapRequired("name", run.name);
    io.mapRequired("pipeline", run.pipeline);
    io.mapOptional("threads-per-group", run.threadsPerThreadgroup);
    io.mapOptional("arguments", run.arguments);
  }
};
template <> struct MappingTraits<sconeml::test::Config> {
  static void mapping(IO &io, sconeml::test::Config &config) {
    io.mapRequired("version", config.version);
    io.mapRequired("input", config.input);
    io.mapRequired("runs", config.runs);
    io.mapOptional("tensor-input", config.tensorInput);
  }
};
template <> struct MappingTraits<sconeml::test::ExpectedError> {
  static void mapping(IO &io, sconeml::test::ExpectedError &error) {
    io.mapRequired("stage", error.stage);
    io.mapOptional("op", error.op);
    io.mapRequired("message", error.message);
  }
};
} // namespace llvm::yaml

namespace sconeml::test {
void require(bool condition, const std::string &message) {
  if (!condition)
    throw std::runtime_error(message);
}

std::string trim(const std::string &text) {
  return llvm::StringRef(text).trim().str();
}

template <typename T> T yaml(const std::string &text) {
  std::string diagnostic;
  llvm::yaml::Input input(
      text, nullptr,
      [](const llvm::SMDiagnostic &diag, void *context) {
        llvm::raw_string_ostream stream(*static_cast<std::string *>(context));
        diag.print("test config", stream);
      },
      &diagnostic);
  T value;
  input >> value;
  require(!input.error(), "invalid YAML: " + diagnostic);
  return value;
}

bool knownPipeline(const std::string &name) {
  static const std::set<std::string> names{
      "verify",         "letalg",        "letalg-opt",          "tensor-scf",
      "tensor-effects", "tensor-gpu-ir", "tensor-msl",          "metal-compile",
      "tensor-cpu",     "scalar-cpu",    "tensor-metal",        "legacy-letalg",
      "legacy-cpu", "tensor-cpu-ir", "tensor-metal-ir",
      "tensor-input-cpu", "tensor-input-ir"};
  return names.contains(name);
}

static bool numericPipeline(const std::string &pipeline) {
  return pipeline == "tensor-cpu" || pipeline == "scalar-cpu" ||
         pipeline == "legacy-cpu" || pipeline == "tensor-metal" ||
         pipeline == "tensor-cpu-ir" || pipeline == "tensor-metal-ir" ||
         pipeline == "tensor-input-cpu";
}

static bool supports(const std::string &pipeline, const std::string &kind) {
  if (kind == "entrypoint")
    return pipeline == "tensor-msl" || pipeline == "metal-compile" ||
           pipeline == "tensor-metal-ir";
  if (kind == "success" || kind == "error")
    return true;
  if (kind == "result" || kind == "compare")
    return numericPipeline(pipeline);
  if (kind == "effects")
    return pipeline == "tensor-effects";
  if (kind == "source-unchanged")
    return pipeline == "tensor-gpu-ir" || pipeline == "tensor-msl" ||
           pipeline == "metal-compile" || pipeline == "tensor-metal-ir";
  if (kind == "ir-check" || kind == "ir-equal")
    return pipeline == "letalg" || pipeline == "letalg-opt" ||
           pipeline == "legacy-letalg" || pipeline == "tensor-scf" ||
           pipeline == "tensor-gpu-ir" || pipeline == "tensor-msl" ||
           pipeline == "metal-compile" || pipeline == "tensor-cpu" ||
           pipeline == "tensor-cpu-ir" || pipeline == "tensor-metal-ir" ||
           pipeline == "tensor-input-ir" || pipeline == "tensor-input-cpu";
  return pipeline == "letalg" || pipeline == "letalg-opt" ||
         pipeline == "legacy-letalg" || pipeline == "tensor-msl" ||
         pipeline == "metal-compile" || pipeline == "tensor-metal" ||
         pipeline == "tensor-metal-ir" || pipeline == "tensor-input-ir";
}

ExpectedError parseError(const std::string &text) {
  auto error = yaml<ExpectedError>(text);
  static const std::set<std::string> stages{
      "parse",        "frontend",     "verify",     "optimize",
      "cpu-lowering", "gpu-lowering", "msl-export", "metal-compile",
      "cpu-execute",  "metal-execute"};
  require(stages.contains(error.stage) && !error.message.empty(),
          "error assertion needs a known stage and a nonempty message");
  return error;
}

Number parseNumber(const std::string &text) {
  auto value = trim(text);
  require(!value.empty(), "empty numeric result");
  Number number;
  std::vector<int64_t> values;
  if (value.front() == '[') {
    number.tensor = true;
    values = yaml<std::vector<int64_t>>(value);
  } else {
    int64_t parsed = 0;
    auto [end, error] =
        std::from_chars(value.data(), value.data() + value.size(), parsed);
    require(error == std::errc{} && end == value.data() + value.size(),
            "result must be an i32 integer or array");
    values.push_back(parsed);
  }
  for (int64_t element : values) {
    require(element >= std::numeric_limits<int32_t>::min() &&
                element <= std::numeric_limits<int32_t>::max(),
            "result outside i32 range");
    number.values.push_back(static_cast<int32_t>(element));
  }
  return number;
}

std::string formatNumber(const Number &number) {
  std::string text = number.tensor ? "[" : "";
  for (size_t i = 0; i < number.values.size(); ++i) {
    if (i)
      text += ", ";
    text += std::to_string(number.values[i]);
  }
  return text + (number.tensor ? "]" : "");
}

TestCase parseCase(const std::string &path, const std::string &text) {
  std::vector<std::pair<std::string, std::string>> blocks;
  size_t position = 0;
  const bool ir = llvm::StringRef(path).ends_with(".mlir");
  if (ir) {
    std::string marker, payload;
    auto flush = [&] {
      require(!marker.empty(), "@end without an annotation");
      blocks.emplace_back(marker, trim(payload));
      marker.clear();
      payload.clear();
    };
    while (position < text.size()) {
      auto end = text.find('\n', position);
      if (end == std::string::npos)
        end = text.size();
      auto line = llvm::StringRef(text).slice(position, end).ltrim();
      if (!line.empty() && !line.starts_with("//"))
        break;
      if (line.consume_front("//")) {
        line.consume_front(" ");
        auto stripped = line.trim();
        if (stripped == "@end")
          flush();
        else if (stripped.starts_with("@")) {
          require(marker.empty(), "missing @end before " + stripped.str());
          marker = stripped.str();
        } else if (!marker.empty())
          payload += line.str() + "\n";
      }
      position = end == text.size() ? end : end + 1;
    }
    require(marker.empty(), "unterminated annotation; expected // @end");
  } else {
    while (position < text.size()) {
      position = text.find_first_not_of(" \t\r\n", position);
      if (position == std::string::npos) {
        position = text.size();
        break;
      }
      if (text.compare(position, 2, "(*") != 0)
        break;
      auto end = text.find("*)", position + 2);
      require(end != std::string::npos, "unterminated annotation comment");
      auto content = trim(text.substr(position + 2, end - position - 2));
      if (!content.empty() && content.front() == '@') {
        auto newline = content.find('\n');
        blocks.emplace_back(trim(content.substr(0, newline)),
                            newline == std::string::npos
                                ? ""
                                : trim(content.substr(newline + 1)));
      }
      position = end + 2;
    }
  }
  TestCase test;
  test.path = path;
  // Keep line offsets for MLIR diagnostics while removing annotation payload.
  test.program =
      std::string(std::count(text.begin(), text.begin() + position, '\n'),
                  '\n') +
      text.substr(position);
  require(!trim(test.program).empty(), "missing test program");
  std::optional<Config> config;
  bool legacy = false;
  for (const auto &[header, payload] : blocks) {
    if (header == "@test") {
      require(!config, "duplicate @test configuration");
      config = yaml<Config>(payload);
    } else if (header == "@letalg:opt" || header == "@runner:cpu") {
      legacy = true;
      auto name = header == "@letalg:opt" ? "legacy-ir" : "legacy-cpu";
      require(!test.assertions.contains(name), "duplicate legacy annotation");
      test.runs.push_back(
          {name, header == "@letalg:opt" ? "legacy-letalg" : "legacy-cpu"});
      test.assertions[name].push_back(
          {header == "@letalg:opt" ? "text-equal" : "result", payload});
    } else if (llvm::StringRef(header).starts_with("@assert:")) {
      auto rest = llvm::StringRef(header).drop_front(8);
      auto [name, kind] = rest.split(':');
      require(!name.empty() && !kind.empty() && !kind.contains(':'),
              "invalid assert header: " + header);
      auto &assertions = test.assertions[name.str()];
      for (const auto &assertion : assertions)
        require(assertion.kind != kind, "duplicate assertion: " + header);
      assertions.push_back({kind.str(), payload});
    } else
      require(false, "unknown annotation: " + header);
  }
  require(!(config && legacy), "cannot mix legacy and @test annotations");
  if (config) {
    require(config->version == 1, "unsupported test format version");
    require(config->input == (ir ? "mlir" : "source"),
            "input must match the .ml/.mlir file kind");
    test.input = config->input;
    test.runs = config->runs;
    if (config->tensorInput) {
      require(ir, "tensor-input requires mlir input");
      test.tensorInput.emplace();
      for (int64_t value : *config->tensorInput) {
        require(value >= std::numeric_limits<int32_t>::min() &&
                    value <= std::numeric_limits<int32_t>::max(),
                "tensor-input value outside i32 range");
        test.tensorInput->push_back(static_cast<int32_t>(value));
      }
    }
  } else {
    require(legacy && !ir, "missing @test configuration or legacy assertions");
    test.input = "source";
  }
  require(!test.runs.empty(), "empty run list");
  std::set<std::string> names;
  std::map<std::string, std::string> preceding;
  bool consumesInput = false;
  for (const auto &run : test.runs) {
    require(!run.name.empty() && names.insert(run.name).second,
            "empty or duplicate run name: " + run.name);
    require(knownPipeline(run.pipeline), "unknown pipeline: " + run.pipeline);
    const bool sourceInputs = run.pipeline == "tensor-input-cpu" ||
                              run.pipeline == "tensor-input-ir";
    require(!sourceInputs || (test.input == "source" && run.arguments.has_value()),
            "external-input pipeline requires source input and arguments");
    require(!run.arguments || sourceInputs,
            "arguments is only supported by external-input pipelines");
    const bool mapIR = run.pipeline == "tensor-cpu-ir" ||
                       run.pipeline == "tensor-metal-ir";
    if (mapIR) {
      require(ir && test.tensorInput.has_value(),
              "IR execution requires mlir input and tensor-input");
      consumesInput = true;
    }
    require(!run.threadsPerThreadgroup || run.pipeline == "tensor-metal" ||
                run.pipeline == "tensor-metal-ir",
            "threads-per-group is only supported by tensor-metal pipelines");
    require(!run.threadsPerThreadgroup || *run.threadsPerThreadgroup > 0,
            "threads-per-group must be positive");
    require(!run.threadsPerThreadgroup ||
                *run.threadsPerThreadgroup <= std::numeric_limits<std::size_t>::max(),
            "threads-per-group exceeds host size range");
    if (run.pipeline == "tensor-cpu" || run.pipeline == "tensor-metal" ||
        run.pipeline == "scalar-cpu" || run.pipeline.starts_with("legacy-"))
      require(test.input == "source",
              "execution/legacy pipeline requires source input");
    auto found = test.assertions.find(run.name);
    require(found != test.assertions.end() && !found->second.empty(),
            "run has no assertions: " + run.name);
    bool error = false;
    for (const auto &assertion : found->second) {
      static const std::set<std::string> kinds{
          "result",   "compare",          "error",      "success",
          "ir-equal", "ir-check",         "text-equal", "text-check",
          "effects",  "source-unchanged", "entrypoint"};
      require(kinds.contains(assertion.kind),
              "unknown assertion: " + assertion.kind);
      require(supports(run.pipeline, assertion.kind),
              assertion.kind + " not supported by " + run.pipeline);
      require(!assertion.value.empty(), "empty assertion: " + assertion.kind);
      if (assertion.kind == "error") {
        parseError(assertion.value);
        error = true;
      }
      if (assertion.kind == "result")
        parseNumber(assertion.value);
      if (assertion.kind == "compare")
        require(preceding.contains(assertion.value) &&
                    numericPipeline(preceding.at(assertion.value)),
                "compare must reference an earlier run: " + assertion.value);
      if (assertion.kind == "success" || assertion.kind == "source-unchanged")
        require(assertion.value == "true", assertion.kind + " must be true");
    }
    require(!error || found->second.size() == 1,
            "error cannot be combined with success assertions");
    preceding.emplace(run.name, run.pipeline);
  }
  require(!test.tensorInput || consumesInput,
          "tensor-input requires an IR execution run");
  for (const auto &[name, assertions] : test.assertions)
    require(names.contains(name), "assertion refers to unknown run: " + name);
  return test;
}

TestCase readCase(const std::string &path) {
  auto buffer = llvm::MemoryBuffer::getFile(path);
  require(bool(buffer), "cannot read test file: " + path);
  return parseCase(path, (*buffer)->getBuffer().str());
}
} // namespace sconeml::test
