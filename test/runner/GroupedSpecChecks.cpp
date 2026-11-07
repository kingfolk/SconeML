#include "test/runner/TestSpec.h"
#include <algorithm>
#include <iostream>
#include <stdexcept>

using namespace sconeml::test;

namespace {
const std::string spec = R"((* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: tensor-input-cpu
    arguments:
      - name: a
        values: [2, 3]
*)
(* @assert:cpu:result
[3, 4]
*)
)";

template <class F> void rejects(F action, const std::string &message) {
  try {
    action();
  } catch (const std::exception &error) {
    require(std::string(error.what()).find(message) != std::string::npos,
            "wrong rejection: " + std::string(error.what()));
    return;
  }
  throw std::runtime_error("expected rejection: " + message);
}
} // namespace

int main(int argc, char **argv) {
  try {
    const std::string first = "(* @case add *)\n" + spec + "a + 1\n";
    const std::string second = "(* @case square *)\n" + spec + "a * a\n";
    auto tests = parseCases("group.ml", first + second);
    require(tests.size() == 2 && tests[0].name == "add" &&
                tests[1].name == "square",
            "lost grouped case");
    require(tests[1].id() == "group.ml#square", "missing case identity");
    const auto combined = first + second;
    const auto programStart = combined.find("a * a");
    require(tests[1].programLine ==
                1 + std::count(combined.begin(),
                               combined.begin() + programStart, '\n'),
            "physical source line lost");
    require(
        std::count(tests[1].program.begin(),
                   tests[1].program.begin() + tests[1].program.find("a * a"),
                   '\n') == tests[1].programLine - 1,
        "source prefix must retain blank physical lines");
    rejects([&] { parseCase("group.ml", first + second); }, "explicit case");
    require(parseCase("group.ml", first + second, "square").name == "square",
            "single-case selector ignored");
    rejects([&] { parseCase("group.ml", first + second, "absent"); },
            "case not found");
    rejects([&] { parseCases("group.ml", first + first); }, "duplicate case");
    rejects([&] { parseCases("group.ml", "a + 1\n" + first); },
            "precede first");
    rejects([&] { parseCases("group.ml", spec + first); }, "precede first");
    rejects([&] { parseCases("group.ml", first + "(* @case empty *)\n"); },
            "group.ml#empty: missing test program");
    rejects(
        [&] {
          parseCases("group.ml", first + "(* @case comments *)\n" + spec +
                                     "(* no executable program *)\n");
        },
        "missing test program");
    for (auto marker :
         {"(* @case *)", "(* @case bad.name *)", "(* @casebad *)",
          "(* @case a b *)", "(* @case a\nb *)", "(* @case inline *) a + 1"})
      rejects(
          [&] {
            parseCases("group.ml", std::string(marker) + "\n" + spec + "a");
          },
          "malformed @case");
    rejects([&] { parseCases("group.ml", first + "(* @case bad *)\n" + spec); },
            "group.ml#bad");
    rejects([&] { parseCases("group.ml", "(* unclosed\n" + first); },
            "unterminated annotation comment at line 1");

    // These strings test fixture discovery only; they are not source-language
    // additions. A production parser may still reject the opaque program text.
    rejects(
        [&] {
          parseCases("group.ml", "(* @case legacy *)\n"
                                 "(* @runner:cpu\n1\n*)\n1");
        },
        "version-1 @test");
    const auto opaque = spec + "\"escaped \\\"\n(* @case fake *)\n\"\na + 1\n";
    require(parseCases("group.ml", opaque).size() == 1,
            "split inside a string");
    const auto comments = "(* ordinary\n(* @case fake *)\n*)\n";
    require(parseCases("group.ml", comments + first + second).size() == 2,
            "split inside a nested comment");
    const auto annotation = R"((* @test
version: 1
input: source
runs:
  - name: ir
    pipeline: tensor-input-ir
    arguments: []
*)
(* @assert:ir:text-check
CHECK: (* @case payload *)
*)
a
)";
    require(
        parseCases("group.ml", "(* @case real *)\n" + std::string(annotation))
                .size() == 1,
        "split inside an annotation payload");
    require(parseCases("single.ml", spec + "a + 1").front().name.empty(),
            "ungrouped compatibility lost");
    require(argc == 2, "fixture path required");
    require(readCases(argv[1]).size() == 22, "reader dropped grouped cases");
    rejects([&] { readCase(argv[1]); }, "explicit case");
    require(readCase(argv[1], "w4-n005").name == "w4-n005",
            "file reader selection ignored");
    std::cout
        << "Grouped fixture lexical, isolation and reader contracts passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
