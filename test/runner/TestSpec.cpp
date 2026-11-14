#include "test/runner/TestSpec.h"

#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Support/raw_ostream.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/Support/Error.h"
#include <algorithm>
#include <bit>
#include <charconv>
#include <iomanip>
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
    io.mapOptional("dtype", argument.dtype);
    if (argument.dtype != "i32" && argument.dtype != "f32")
      io.setError("unsupported argument dtype");
    if (argument.dtype == "f32")
      io.mapRequired("values", argument.f32Values);
    else
      io.mapRequired("values", argument.values);
    io.mapOptional("shape", argument.shape);
  }
};
template <> struct MappingTraits<sconeml::test::Run> {
  static void mapping(IO &io, sconeml::test::Run &run) {
    io.mapRequired("name", run.name);
    io.mapRequired("pipeline", run.pipeline);
    io.mapOptional("threads-per-group", run.threadsPerThreadgroup);
    io.mapOptional("arguments", run.arguments);
    io.mapOptional("profile", run.profile);
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
  static const std::set<std::string> names{"verify",
                                           "letalg",
                                           "letalg-opt",
                                           "tensor-scf",
                                           "tensor-effects",
                                           "tensor-gpu-ir",
                                           "tensor-msl",
                                           "metal-compile",
                                           "tensor-cpu",
                                           "scalar-cpu",
                                           "tensor-metal",
                                           "legacy-letalg",
                                           "legacy-cpu",
                                           "tensor-cpu-ir",
                                           "tensor-metal-ir",
                                           "tensor-input-cpu",
                                           "tensor-input-ir",
                                           "tensor-input-metal",
                                           "shaped-input-ir",
                                           "shaped-input-cpu",
                                           "shaped-input-metal",
                                           "shaped-input-gpu-ir",
                                           "shaped-input-msl",
                                           "shaped-f32-cpu",
                                           "shaped-f32-ir",
                                           "shaped-f32-metal",
                                           "shaped-f32-gpu-ir",
                                           "shaped-f32-msl",
                                           "sum-i32-cpu",
                                           "sum-f32-cpu",
                                           "sum-i32-ir",
                                           "sum-f32-ir"};
  return names.contains(name);
}

static bool numericPipeline(const std::string &pipeline) {
  if (pipeline.starts_with("sum-"))
    return pipeline.ends_with("-cpu") || pipeline.ends_with("-metal");
  return pipeline == "tensor-cpu" || pipeline == "scalar-cpu" ||
         pipeline == "legacy-cpu" || pipeline == "tensor-metal" ||
         pipeline == "tensor-cpu-ir" || pipeline == "tensor-metal-ir" ||
         pipeline == "tensor-input-cpu" || pipeline == "tensor-input-metal" ||
         pipeline == "shaped-input-cpu" || pipeline == "shaped-input-metal" ||
         (pipeline == "shaped-f32-cpu" || pipeline == "shaped-f32-metal");
}

static bool supports(const std::string &pipeline, const std::string &kind) {
  if (pipeline.starts_with("sum-")) {
    if (kind == "success" || kind == "error")
      return true;
    if (kind == "result" || kind == "shape" || kind == "compare")
      return numericPipeline(pipeline);
    if (kind == "ir-check" || kind == "ir-equal")
      return pipeline.ends_with("-ir") || pipeline.ends_with("-cpu");
    return false;
  }
  if (pipeline.starts_with("shaped-f32-")) {
    if (kind == "success" || kind == "error")
      return true;
    if (kind == "ir-check" || kind == "ir-equal")
      return pipeline != "shaped-f32-metal";
    if (kind == "text-check" || kind == "text-equal")
      return pipeline == "shaped-f32-msl" || pipeline == "shaped-f32-metal";
    if (kind == "source-unchanged")
      return pipeline == "shaped-f32-gpu-ir" || pipeline == "shaped-f32-msl";
    if (kind == "entrypoint")
      return pipeline == "shaped-f32-msl";
    return (pipeline == "shaped-f32-cpu" || pipeline == "shaped-f32-metal") &&
           (kind == "result" || kind == "shape" || kind == "compare");
  }
  if (kind == "shape")
    return pipeline == "shaped-input-cpu" || pipeline == "shaped-input-metal" ||
           (pipeline == "shaped-f32-cpu" || pipeline == "shaped-f32-metal");
  if ((kind == "ir-check" || kind == "ir-equal") &&
      (pipeline == "shaped-input-ir" || pipeline == "shaped-input-cpu" ||
       pipeline == "shaped-input-gpu-ir" || pipeline == "shaped-input-msl"))
    return true;
  if ((kind == "text-check" || kind == "text-equal") &&
      (pipeline == "shaped-input-msl" || pipeline == "shaped-input-metal")) return true;
  if (kind == "source-unchanged" &&
      (pipeline == "shaped-input-gpu-ir" || pipeline == "shaped-input-msl")) return true;
  if (kind == "entrypoint" && pipeline == "shaped-input-msl") return true;
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
         pipeline == "tensor-metal-ir" || pipeline == "tensor-input-ir" ||
         pipeline == "tensor-input-metal";
}

std::vector<int64_t> parseShape(const std::string &text, bool scalar) {
  auto shape = yaml<std::vector<int64_t>>(text);
  require((scalar || !shape.empty()) && shape.size() <= 2 &&
              std::all_of(shape.begin(), shape.end(),
                          [](int64_t n) { return n >= 0; }),
          "shape assertion requires rank 1 or 2 nonnegative extents");
  return shape;
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

uint32_t parseF32Bits(const std::string &text) {
  auto spelling = trim(text);
  if (spelling.starts_with("0x")) {
    uint32_t bits;
    auto [end, error] = std::from_chars(
        spelling.data() + 2, spelling.data() + spelling.size(), bits, 16);
    require(error == std::errc{} && end == spelling.data() + spelling.size(),
            "invalid f32 bit pattern");
    return bits;
  }
  if (spelling == "inf" || spelling == "+inf")
    return 0x7f800000u;
  if (spelling == "-inf")
    return 0xff800000u;
  if (spelling == "nan")
    return 0x7fc00000u;
  llvm::APFloat value(llvm::APFloat::IEEEsingle());
  auto status =
      value.convertFromString(spelling, llvm::APFloat::rmNearestTiesToEven);
  if (!status) {
    llvm::consumeError(status.takeError());
    throw std::runtime_error("invalid f32 data value");
  }
  require(value.isFinite(), "f32 data overflow; use explicit inf or raw bits");
  return static_cast<uint32_t>(value.bitcastToAPInt().getZExtValue());
}
std::vector<float> f32Values(const InputArgument &argument) {
  std::vector<float> result;
  for (const auto &text : argument.f32Values)
    result.push_back(std::bit_cast<float>(parseF32Bits(text)));
  return result;
}
bool Number::operator==(const Number &other) const {
  if (tensor != other.tensor || f32 != other.f32)
    return false;
  if (!f32)
    return values == other.values;
  if (f32Bits.size() != other.f32Bits.size())
    return false;
  auto nan = [](uint32_t b) { return (b & 0x7fffffffu) > 0x7f800000u; };
  for (size_t i = 0; i < f32Bits.size(); ++i)
    if (f32Bits[i] != other.f32Bits[i] &&
        !(nan(f32Bits[i]) && nan(other.f32Bits[i])))
      return false;
  return true;
}

Number parseNumber(const std::string &text, bool f32) {
  if (f32) {
    Number number;
    number.tensor = true;
    number.f32 = true;
    require(trim(text).starts_with("["), "f32 result requires an array");
    for (const auto &element : yaml<std::vector<std::string>>(text))
      number.f32Bits.push_back(parseF32Bits(element));
    return number;
  }
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
  if (number.f32) {
    std::ostringstream stream;
    stream << "[";
    for (size_t i = 0; i < number.f32Bits.size(); ++i) {
      if (i)
        stream << ", ";
      stream << "0x" << std::hex << std::setw(8) << std::setfill('0')
             << number.f32Bits[i];
    }
    return stream.str() + "]";
  }
  std::string text = number.tensor ? "[" : "";
  for (size_t i = 0; i < number.values.size(); ++i) {
    if (i)
      text += ", ";
    text += std::to_string(number.values[i]);
  }
  return text + (number.tensor ? "]" : "");
}

namespace {
// Scan source comments once per level. Nested comments and escaped strings are
// opaque to case discovery; annotation payloads are comments too.
size_t commentEnd(const std::string &text, size_t start) {
  size_t depth = 1, position = start + 2;
  while (position < text.size()) {
    if (text.compare(position, 2, "(*") == 0) {
      ++depth;
      position += 2;
    } else if (text.compare(position, 2, "*)") == 0) {
      position += 2;
      if (--depth == 0)
        return position;
    } else
      ++position;
  }
  throw std::runtime_error(
      "unterminated annotation comment at line " +
      std::to_string(1 + std::count(text.begin(), text.begin() + start, '\n')));
}

bool sourceHasCode(const std::string &text) {
  size_t position = 0;
  while (position < text.size()) {
    if (text.compare(position, 2, "(*") == 0)
      position = commentEnd(text, position);
    else if (llvm::StringRef(text).substr(position, 1).trim().empty())
      ++position;
    else
      return true;
  }
  return false;
}
} // namespace

static TestCase parseSingleCase(const std::string &path,
                                const std::string &text) {
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
      auto end = commentEnd(text, position) - 2;
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
  test.programLine =
      1 + std::count(text.begin(), text.begin() + position, '\n');
  require(ir ? !trim(test.program).empty() : sourceHasCode(test.program),
          "missing test program");
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
    const bool floatRun = run.pipeline.starts_with("shaped-f32-") ||
                          run.pipeline.starts_with("sum-f32-");
    const bool sumRun = run.pipeline.starts_with("sum-");
    const bool shaped =
        run.pipeline.starts_with("shaped-input-") || floatRun || sumRun;
    require(floatRun
                ? (run.profile &&
                   *run.profile == (sumRun ? "f32-sum-tree256-ftz" : "f32-ftz"))
                : !run.profile,
            "f32 runs require explicit f32-ftz profile; profile forbidden on "
            "i32 runs");
    const bool sourceInputs = shaped || run.pipeline == "tensor-input-cpu" ||
                              run.pipeline == "tensor-input-ir" ||
                              run.pipeline == "tensor-input-metal";
    require(!sourceInputs ||
                (test.input == "source" && run.arguments.has_value()),
            "external-input pipeline requires source input and arguments");
    require(!run.arguments || sourceInputs,
            "arguments is only supported by external-input pipelines");
    if (run.arguments)
      for (const auto &argument : *run.arguments) {
        require(argument.shape.has_value() == shaped,
                "shape is required for shaped pipelines and forbidden for flat pipelines");
        require(argument.dtype == (floatRun ? "f32" : "i32"),
                "argument dtype does not match pipeline");
        if (floatRun)
          f32Values(argument);
      }
    const bool mapIR =
        run.pipeline == "tensor-cpu-ir" || run.pipeline == "tensor-metal-ir";
    if (mapIR) {
      require(ir && test.tensorInput.has_value(),
              "IR execution requires mlir input and tensor-input");
      consumesInput = true;
    }
    require(!run.threadsPerThreadgroup || run.pipeline == "tensor-metal" ||
                run.pipeline == "tensor-metal-ir" ||
                run.pipeline == "tensor-input-metal" ||
                (run.pipeline == "shaped-input-metal" ||
                 run.pipeline == "shaped-f32-metal"),
            "threads-per-group is only supported by tensor-metal pipelines");
    require(!run.threadsPerThreadgroup || *run.threadsPerThreadgroup > 0,
            "threads-per-group must be positive");
    require(!run.threadsPerThreadgroup ||
                *run.threadsPerThreadgroup <=
                    std::numeric_limits<std::size_t>::max(),
            "threads-per-group exceeds host size range");
    if (run.pipeline == "tensor-cpu" || run.pipeline == "tensor-metal" ||
        run.pipeline == "scalar-cpu" || run.pipeline.starts_with("legacy-"))
      require(test.input == "source",
              "execution/legacy pipeline requires source input");
    auto found = test.assertions.find(run.name);
    require(found != test.assertions.end() && !found->second.empty(),
            "run has no assertions: " + run.name);
    bool error = false, hasShape = false, hasValues = false;
    for (const auto &assertion : found->second) {
      static const std::set<std::string> kinds{
          "result",   "compare",          "error",      "success",
          "ir-equal", "ir-check",         "text-equal", "text-check",
          "effects",  "source-unchanged", "entrypoint", "shape"};
      require(kinds.contains(assertion.kind),
              "unknown assertion: " + assertion.kind);
      require(supports(run.pipeline, assertion.kind),
              assertion.kind + " not supported by " + run.pipeline);
      require(!assertion.value.empty(), "empty assertion: " + assertion.kind);
      if (assertion.kind == "error") {
        parseError(assertion.value);
        error = true;
      }
      if (assertion.kind == "shape") {
        parseShape(assertion.value, sumRun);
        hasShape = true;
      }
      hasValues |= assertion.kind == "result" || assertion.kind == "compare";
      if (assertion.kind == "result")
        parseNumber(assertion.value, floatRun);
      if (assertion.kind == "compare")
        require(
            preceding.contains(assertion.value) &&
                numericPipeline(preceding.at(assertion.value)) &&
                floatRun ==
                    (preceding.at(assertion.value).starts_with("shaped-f32-") ||
                     preceding.at(assertion.value).starts_with("sum-f32-")),
            "compare must reference an earlier run: " + assertion.value);
      if (assertion.kind == "success" || assertion.kind == "source-unchanged")
        require(assertion.value == "true", assertion.kind + " must be true");
    }
    require(!error || found->second.size() == 1,
            "error cannot be combined with success assertions");
    const bool shapedExecution = (sumRun && numericPipeline(run.pipeline)) ||
                                 run.pipeline == "shaped-input-cpu" ||
                                 run.pipeline == "shaped-input-metal" ||
                                 (run.pipeline == "shaped-f32-cpu" ||
                                  run.pipeline == "shaped-f32-metal");
    require(error || !shapedExecution || (hasShape && hasValues),
            "shaped execution requires shape and result or compare assertions: " + run.name);
    preceding.emplace(run.name, run.pipeline);
  }
  require(!test.tensorInput || consumesInput,
          "tensor-input requires an IR execution run");
  for (const auto &[name, assertions] : test.assertions)
    require(names.contains(name), "assertion refers to unknown run: " + name);
  return test;
}

std::vector<TestCase> parseCases(const std::string &path,
                                 const std::string &text) {
  if (llvm::StringRef(path).ends_with(".mlir"))
    return {parseSingleCase(path, text)};
  struct Marker {
    size_t start, end;
    std::string name;
  };
  std::vector<Marker> markers;
  std::set<std::string> names;
  bool preamble = true;
  for (size_t position = 0; position < text.size();) {
    if (text.compare(position, 2, "(*") == 0) {
      const size_t end = commentEnd(text, position);
      const auto content = trim(text.substr(position + 2, end - position - 4));
      if (llvm::StringRef(content).starts_with("@case")) {
        const auto name = trim(content.substr(5));
        const size_t lineStart = text.rfind('\n', position);
        const size_t lineEnd = text.find('\n', end);
        const bool standalone =
            trim(text.substr(lineStart == std::string::npos ? 0 : lineStart + 1,
                             position - (lineStart == std::string::npos
                                             ? 0
                                             : lineStart + 1)))
                .empty() &&
            trim(text.substr(end, lineEnd == std::string::npos ? lineEnd
                                                               : lineEnd - end))
                .empty();
        const bool validName =
            !name.empty() &&
            std::all_of(name.begin(), name.end(), [](unsigned char c) {
              return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                     (c >= '0' && c <= '9') || c == '_' || c == '-';
            });
        const auto context =
            path + "#" + name + ": line " +
            std::to_string(
                1 + std::count(text.begin(), text.begin() + position, '\n'));
        require(standalone && content.size() > 5 &&
                    (content[5] == ' ' || content[5] == '\t') &&
                    content.find('\n') == std::string::npos && validName,
                context + ": malformed @case marker");
        require(!markers.empty() || preamble,
                context + ": only whitespace and ordinary comments may precede "
                          "first @case");
        require(names.insert(name).second, context + ": duplicate case name");
        markers.push_back({position, end, name});
      } else if (!content.empty() && content.front() == '@' && markers.empty())
        preamble = false;
      position = end;
    } else if (text[position] == '"') {
      preamble = false;
      ++position;
      while (position < text.size()) {
        if (text[position] == '\\')
          position += std::min(size_t(2), text.size() - position);
        else if (text[position++] == '"')
          break;
      }
    } else {
      if (!llvm::StringRef(text).substr(position, 1).trim().empty())
        preamble = false;
      ++position;
    }
  }
  if (markers.empty())
    return {parseSingleCase(path, text)};
  std::vector<TestCase> tests;
  for (size_t i = 0; i < markers.size(); ++i) {
    const auto &marker = markers[i];
    const size_t end =
        i + 1 < markers.size() ? markers[i + 1].start : text.size();
    // Prefix blank lines preserve physical source positions for every consumer.
    auto chunk =
        std::string(std::count(text.begin(), text.begin() + marker.end, '\n'),
                    '\n') +
        text.substr(marker.end, end - marker.end);
    try {
      auto test = parseSingleCase(path, chunk);
      require(std::none_of(test.runs.begin(), test.runs.end(),
                           [](const Run &run) {
                             return run.pipeline.starts_with("legacy-");
                           }),
              "named cases require version-1 @test configuration");
      test.name = marker.name;
      tests.push_back(std::move(test));
    } catch (const std::exception &error) {
      throw std::runtime_error(path + "#" + marker.name + ": " + error.what());
    }
  }
  return tests;
}

static TestCase selectCase(std::vector<TestCase> tests,
                           const std::string &name) {
  if (name.empty()) {
    require(tests.size() == 1,
            "multi-case file requires an explicit case selection");
    return std::move(tests.front());
  }
  for (auto &test : tests)
    if (test.name == name)
      return std::move(test);
  throw std::runtime_error("case not found: " + name);
}

TestCase parseCase(const std::string &path, const std::string &text,
                   const std::string &name) {
  return selectCase(parseCases(path, text), name);
}

std::vector<TestCase> readCases(const std::string &path) {
  auto buffer = llvm::MemoryBuffer::getFile(path);
  require(bool(buffer), "cannot read test file: " + path);
  return parseCases(path, (*buffer)->getBuffer().str());
}

TestCase readCase(const std::string &path, const std::string &name) {
  return selectCase(readCases(path), name);
}
} // namespace sconeml::test
