#include "test/runner/Pipelines.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <set>
#ifdef _WIN32
#include <io.h>
#define SCONEML_DUP _dup
#define SCONEML_DUP2 _dup2
#define SCONEML_CLOSE _close
#define SCONEML_FILENO _fileno
#else
#include <unistd.h>
#define SCONEML_DUP dup
#define SCONEML_DUP2 dup2
#define SCONEML_CLOSE close
#define SCONEML_FILENO fileno
#endif

namespace fs = std::filesystem;
using namespace sconeml::test;

namespace {
// The existing parser uses printf. Capture C stdout at the descriptor level
// so successful runs stay quiet without changing production parsing behavior.
struct CaptureTrace {
  explicit CaptureTrace(std::string &text) : text(text) {
    llvm::outs().flush();
    std::fflush(stdout);
    file = std::tmpfile();
    require(file != nullptr, "cannot create parser trace buffer");
    saved = SCONEML_DUP(SCONEML_FILENO(stdout));
    if (saved < 0 ||
        SCONEML_DUP2(SCONEML_FILENO(file), SCONEML_FILENO(stdout)) < 0) {
      if (saved >= 0)
        SCONEML_CLOSE(saved);
      std::fclose(file);
      throw std::runtime_error("cannot capture parser trace");
    }
  }
  ~CaptureTrace() {
    llvm::outs().flush();
    std::fflush(stdout);
    SCONEML_DUP2(saved, SCONEML_FILENO(stdout));
    SCONEML_CLOSE(saved);
    std::rewind(file);
    char buffer[4096];
    while (auto count = std::fread(buffer, 1, sizeof(buffer), file))
      text.append(buffer, count);
    std::fclose(file);
  }
  std::string &text;
  std::FILE *file = nullptr;
  int saved = -1;
};
} // namespace

int main(int argc, char **argv) {
  try {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    std::string root = SCONEML_TEST_ROOT, file, selectedRun;
    bool disableMetal = false, requireMetal = false;
    for (int i = 1; i < argc; ++i) {
      std::string argument = argv[i];
      if (argument == "--file" || argument == "--suite" ||
          argument == "--run") {
        require(i + 1 < argc, "missing value for " + argument);
        (argument == "--file"    ? file
         : argument == "--suite" ? root
                                 : selectedRun) = argv[++i];
      } else if (argument == "--no-metal")
        disableMetal = true;
      else if (argument == "--require-metal")
        requireMetal = true;
      else
        throw std::runtime_error("unknown argument: " + argument);
    }
    std::vector<fs::path> files;
    if (!file.empty() && fs::is_regular_file(file)) {
      files.push_back(fs::absolute(file));
    } else {
      require(fs::is_directory(root), "suite directory not found: " + root);
      for (const auto &entry : fs::recursive_directory_iterator(root)) {
        auto extension = entry.path().extension();
        if (entry.is_regular_file() &&
            (extension == ".ml" || extension == ".mlir") &&
            (file.empty() || entry.path().filename() == file ||
             entry.path().lexically_relative(root) == file))
          files.push_back(entry.path());
      }
      if (!file.empty())
        require(files.size() == 1, "file not found or ambiguous: " + file);
    }
    require(!files.empty(), "empty test suite");
    std::sort(files.begin(), files.end());
    // Validate every selected data file before starting any pipeline/device
    // work.
    std::vector<TestCase> tests;
    for (const auto &path : files) {
      try {
        tests.push_back(readCase(path.string()));
      } catch (const std::exception &error) {
        throw std::runtime_error(path.string() + ": config: " + error.what());
      }
    }
    unsigned passed = 0, skipped = 0, failed = 0, selected = 0;
    for (const auto &test : tests) {
      std::set<std::string> wanted;
      for (const auto &run : test.runs)
        if (selectedRun.empty() || run.name == selectedRun) {
          wanted.insert(run.name);
          ++selected;
        }
      // Include earlier numeric comparison dependencies when selecting one run.
      for (auto it = test.runs.rbegin(); it != test.runs.rend(); ++it)
        if (wanted.contains(it->name))
          for (const auto &assertion : test.assertions.at(it->name))
            if (assertion.kind == "compare")
              wanted.insert(assertion.value);
      std::map<std::string, Output> outputs;
      for (const auto &run : test.runs) {
        if (!wanted.contains(run.name))
          continue;
        Output output;
        std::string trace;
        {
          CaptureTrace capture(trace);
          output = execute(test, run, disableMetal);
        }
        if (!output.success && !output.skipped)
          output.diagnostic += "\n" + trace;
        try {
          for (const auto &assertion : test.assertions.at(run.name))
            if (assertion.kind == "compare" &&
                outputs.at(assertion.value).skipped) {
              output.skipped = true;
              output.diagnostic = "comparison dependency skipped";
            }
          if (output.skipped) {
            require(!requireMetal,
                    "required device unavailable: " + output.diagnostic);
            ++skipped;
            llvm::outs() << "SKIP " << test.path << " [" << run.name
                         << "]: " << output.diagnostic << '\n';
          } else {
            checkAssertions(test, run, output, outputs);
            ++passed;
            llvm::outs() << "PASS " << test.path << " [" << run.name << "]\n";
          }
        } catch (const std::exception &error) {
          ++failed;
          output.success = false;
          llvm::errs() << "FAIL " << test.path << " [" << run.name
                       << "]: " << error.what() << '\n';
        }
        outputs.emplace(run.name, std::move(output));
      }
    }
    require(selected != 0, "run not found: " + selectedRun);
    llvm::outs() << "Tests: " << passed << " passed, " << skipped
                 << " skipped, " << failed << " failed\n";
    return failed ? 1 : 0;
  } catch (const std::exception &error) {
    llvm::errs() << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
