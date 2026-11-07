# M3-T grouped source test acceptance

Accepted locally on November 7, 2026, against the M2-D baseline `c3a7f15`.
This record covers grouped test fixtures and runner behavior. Rank-2 shapes and
broadcasting remain future work.

A source fixture can contain independent version-1 cases introduced by standalone
`(* @case NAME *)` lines. Each case owns its program, run arguments, assertions
and comparison dependencies. `--case` selects an exact name across selected
files; `--run` includes only that case's local comparison dependencies. All
metadata in selected files is validated before execution, including filtered
cases. A failed assertion in a later case fails the command while subsequent
valid cases still execute. Output identifies the absolute file, case and run,
and separately counts files, cases and runs.

The file reader returns all cases. Single-case C++ readers require explicit
selection for multiple cases and validate the entire file. Ungrouped legacy
source and single-module IR fixtures retain their behavior. Nested comments,
annotation payloads and escaped strings cannot introduce case boundaries.
Malformed/duplicate/empty cases, anonymous/named mixtures and cross-case
comparisons fail. Source prefixes preserve original line counts; failures report
the physical program start line. Existing frontends do not consistently provide
precise token locations, and this feature does not add such locations or change
the source grammar.

The later [thematic consolidation](test-layout.md) groups additional source
fixtures and renames the M2 matrix files. The complete-suite physical file count
below records the initial M3-T validation; the current count is 99, with the
same 221 cases and 536 runs.

## Fixture migration and coverage

The 66 generated M2 matrix files are now `matrix/single-input-launch-boundaries.ml`, `matrix/two-input-launch-boundaries.ml` and
`matrix/three-input-launch-boundaries.ml`, with 22 named cases each. The four standalone semantics/lifetime/
overlap files remain for direct API contracts. M2 therefore has seven physical
files, 70 logical cases and 207 runs. Each migrated case payload was compared
against the old generator at `HEAD` (`c3a7f15`): all 66 source/configuration/
assertion payloads and the four standalone files match exactly apart from added
case markers and separating blank lines. No numerical or launch coverage was
removed. Generator checks still verify the exact file set, contents and
independent oracle anchors without writing fixtures.

`grouped_spec_contracts` adds lexical, physical-line and reader API checks without
linking compiler execution code. `runner_contracts` adds independent expressions
with different input arrays, repeated run names, case/run/suite selection,
local comparisons, missing selections, duplicate/empty cases, prevalidation of
excluded bad pipelines, rejected cross-case dependencies, unknown source names,
later assertion failure with continued execution, and strict/empty/frontend-error
device policies. The existing full suite exercises grouped CPU and actual Metal
execution, including explicit and automatic launch widths. Explicit stub runtime
and GPU API contracts remain part of CTest.

## Validation

Environment: Apple M1, arm64, macOS 15.5, LLVM/MLIR 22.1.8, CMake 4.4.2.
Compiler: `/opt/homebrew/opt/llvm@22/bin/clang++`.
The sandbox hides Metal. Real-device runs used the tool permission system to
execute outside that sandbox; no-device and explicit stub runs remain separate.

```sh
cmake -S . -B build
cmake --build build --target tester grouped_spec_checks tensor_interfaces metal_runtime_checks metal_stub_checks cpu_input_contracts gpu_input_contracts gpu_input_stub_contracts -j3
ctest --test-dir build --output-on-failure
python3 test/reference/generate_m2.py --check
python3 -O test/reference/generate_m2.py --check
./build/tester --require-metal
./build/tester --no-metal
./build/tester --suite test/acceptance_m2 --no-metal
./build/tester --file test/acceptance_m2/matrix/three-input-launch-boundaries.ml --case w32-n033 --run metal --require-metal
```

| Check | Observed result |
| --- | --- |
| CTest in sandbox | 11/11 passed |
| CTest with real Metal access, including direct GPU/runtime and explicit stub contracts | 11/11 passed |
| Complete data suite, Metal required | 158 files, 221 logical cases; 536 passed, 0 skipped, 0 failed |
| Complete data suite, Metal disabled | 339 passed, 197 skipped, 0 failed |
| M2 grouped suite, Metal disabled | 7 files, 70 cases, 207 runs; 95 passed, 112 skipped, 0 failed |
| Exact M2 fixture/oracle checks, ordinary and optimized Python | Passed; seven physical files |
| Pre-migration payload comparison | All 66 migrated cases and four standalone fixtures preserved |

The complete 536-run baseline is unchanged. This change reduces fixture file
fragmentation, not execution coverage, and makes no performance claim. The stub
is built and tested explicitly on macOS; this is not a separate Linux validation.
Planning documents remain local and excluded from feature staging.
