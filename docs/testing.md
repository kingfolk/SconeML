# Data-driven compiler tests

`tester` reads each test's program, configuration, and assertions from one file. `.ml` files contain source programs; `.mlir` files contain raw IR. The runner calls production frontend, lowering, exporter, and execution code through a fixed set of pipeline adapters.

The [source test layout](test-layout.md) indexes thematic files, case counts and
selection examples. Related regressions should normally become named cases in
the matching file.

## Running tests

```sh
cmake -S . -B build
cmake --build build --target tester tensor_interfaces metal_runtime_checks metal_stub_checks cpu_input_contracts gpu_input_contracts gpu_input_stub_contracts grouped_spec_checks shaped_cpu_contracts shaped_gpu_contracts shaped_gpu_stub_contracts -j3
ctest --test-dir build --output-on-failure
```

`compiler_suite` runs all compiler data files. `runner_contracts` checks the runner's error handling, selection, and device policy. `tensor_interfaces` retains two C++ API checks: frontend linkage without CPURunner/ExecutionEngine, and rejection of malformed in-memory IR that the text parser cannot construct. Both API tests read their programs from the regular data files. `metal_runtime_contracts` additionally checks synthetic capacity/launch limits and an empty runtime call without allocating large tensors. It also checks compilation/entrypoint-error propagation when a device is available. `metal_stub_contracts` builds the non-Apple implementation explicitly, including on macOS, and checks its empty/nonempty behavior. `reference_fixtures` uses Python 3 to check independent oracle anchors and reproducible fixture contents without rewriting files.

The runner defaults to the project's `test` directory, independent of the working directory. It discovers `.ml` and `.mlir` files recursively in sorted order. Useful selections are:

```sh
./build/tester --file let.ml
./build/tester --file test/tensor/polynomial.ml --run kernel
./build/tester --suite test/semantics
./build/tester --no-metal
./build/tester --run compile --require-metal
```

`--file` accepts an existing path, a unique basename, or a suite-relative path. `--case` selects an exact named source case within those files. `--run` selects an exact run name; comparison dependencies are automatically included. Missing files/cases/runs, ambiguous basenames, empty suites, bad configuration, and failed assertions exit nonzero.

Metal availability is decided per run. A skipped device run does not skip CPU/IR assertions in the same file. `--no-metal` forces the unavailable-device path for testing; `--require-metal` turns device skips into failures. Combining them deliberately tests strict-device failure without requiring special hardware. An unimplemented pipeline is a configuration error, never a device skip.

## Source format

Only leading metadata comments are interpreted. A source case looks like this:

```ocaml
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: tensor-cpu
  - name: kernel
    pipeline: tensor-gpu-ir
*)
(* @assert:cpu:result
[4, 9, 16]
*)
(* @assert:kernel:ir-check
CHECK: gpu.func @tensor_map
CHECK: scf.if
CHECK: memref.load
CHECK: memref.store
*)
let ts = [1, 2, 3] in ts * ts + 2 * ts + 1
```

Every run starts with its own context and newly translated/parsed module. Destructive optimization or lowering in one run cannot affect another.

`version`, `input`, and `runs` are required YAML fields. Each run has `name` and `pipeline`, plus optional positive `threads-per-group` for `tensor-metal` or `tensor-metal-ir`. Omitting the width selects the runtime default. Unknown/duplicate YAML fields, duplicate names or assertion kinds, assertions for unknown runs, empty programs, and missing assertions fail validation before execution. Assertion kinds must be supported by their pipeline. Arbitrary shell commands and pass scripts are not supported.

## Multiple source cases in one file

A standalone top-level `(* @case NAME *)` line starts a named case and ends the
previous one. For example, `arithmetic.ml` can contain:

```ocaml
(* @case add *)
(* @test
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
a + 1

(* @case square *)
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: tensor-input-cpu
    arguments:
      - name: a
        values: [4, 5]
*)
(* @assert:cpu:result
[16, 25]
*)
a * a
```

Each case owns its complete version-1 configuration, source, inputs and
assertions. Repeated run names are allowed across cases. Comparisons reference
only earlier numeric runs in that case; cross-case references and shared source
bindings are unsupported. Every run still creates its own compiler context and
module.

Names are nonempty ASCII letters, digits, underscores or hyphens, unique within
a file. Only whitespace and ordinary comments may precede the first marker.
Anonymous programs/annotations mixed with named blocks, malformed markers,
empty/comment-only programs, missing metadata and duplicate names fail validation.
Discovery respects nested comments, annotation payloads, strings and escaped
quotes; marker-like text inside them does not create a case. This lexical
handling does not extend the production source grammar.

```sh
./build/tester --file arithmetic.ml
./build/tester --file arithmetic.ml --case square --run cpu
./build/tester --suite test/acceptance_m2 --case w32-n033 --run metal --require-metal
./build/tester --file test/acceptance_m2/matrix/three-input-launch-boundaries.ml --case w32-n000 --no-metal --require-metal
```

Case selection matches the same exact name across multiple files in a suite.
Comparison dependencies are automatically included locally. All cases in selected
files are validated before any pipeline runs, even if excluded by `--case`,
`--run` or device policy. Missing selections fail. Execution/assertion failures
identify `absolute/path.ml#case [run]` and the physical source start line; other
valid cases continue. The summary reports selected physical files, logical cases
and actual runs (including comparison dependencies) separately.

Original line prefixes are retained in source strings. Existing source frontends
use unknown MLIR locations and some syntax errors have no token location; the
runner reports the source start line without inventing a precise failing-token
position. Configuration failures include the file/case identity.

Ungrouped source files retain version-1 and legacy behavior. `.mlir` files retain
their single-module format. C++ consumers use `readCases`/`parseCases` for all
cases; `readCase`/`parseCase` reject multiple cases unless an exact case name is
supplied as the optional final argument. They validate the whole file before
selecting. `grouped_spec_contracts` checks lexical discovery, source offsets and
reader behavior without linking a compiler or GPU runtime. `runner_contracts`
checks selection, independent inputs/results, failure continuation, prevalidation,
and strict/empty/error device policies. See [M3-T acceptance](m3-t-acceptance.md).

## Shaped CPU source tests

`shaped-input-ir` checks the production Linalg frontend. `shaped-input-cpu`
executes the public `runShapedTensorCPU` API and supports `result`, `shape`,
`compare` and LLVM `ir-check` assertions. Each argument requires `shape: [R, C]`
or `shape: [N]` alongside its name and row-major values. Shape fields are forbidden
on existing flat pipelines, so a misspelled pipeline cannot ignore shape metadata.
Shaped CPU runs infer trailing-axis broadcasting across all declared inputs,
including unused ones.
`shape` assertions use signed 64-bit extents and require rank 1 or 2. Every
successful `shaped-input-cpu` or `shaped-input-metal` run must include an explicit
`shape` assertion and either `result` or `compare`; `success`, text checks or
flattened values alone cannot satisfy the numeric contract. A sole expected
`error` assertion remains valid. This is checked before selection, device skips
or pipeline execution, including unselected grouped cases. Existing flat test
metadata is unchanged. Numeric comparisons also compare the presence and
contents of output shapes. Semantic
shape errors belong to the frontend and can be asserted with `error`; missing
shape fields are configuration errors before any selected run executes.

```ocaml
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: shaped-input-cpu
    arguments:
      - name: a
        values: [1, 2, 3, 4, 5, 6]
        shape: [2, 3]
*)
(* @assert:cpu:result
[2, 3, 4, 5, 6, 7]
*)
(* @assert:cpu:shape
[2, 3]
*)
a + 1
```

`shaped_cpu_contracts` checks ownership, overlapping input spans, input-value
independence of IR, shape specialization, empty execution, flat API compatibility
and rejection of mutated in-memory Linalg/wrapper IR. The programs and numeric
expectations come from `test/shaped_input/same_shape.ml` and the grouped
broadcast fixtures. `broadcast_reference_fixtures` checks an independent Python
nested-loop oracle with hand-computed anchors and verifies all generated
broadcast fixtures without writing. Regenerate only those fixtures with
`python3 test/reference/generate_broadcast.py`; use `--check` to verify. See the
[API contract](shaped-tensor-cpu.md), [M3-A evidence](m3-a-acceptance.md) and
[M3-B evidence](m3-b-acceptance.md).

## Shaped GPU source tests

`shaped-input-metal` calls the production `runShapedTensorGPU` API and supports
`result`, `shape`, `compare`, MSL text checks and `threads-per-group`.
`shaped-input-gpu-ir` extracts shape-specialized device IR;
`shaped-input-msl` also exports MSL. These device-independent adapters accept
`ir-check` and `source-unchanged`; the MSL adapter also accepts text checks and
`entrypoint`. All require source input with per-argument shape and values.

Shape/storage/syntax errors report at `frontend`; GPU target and launch bounds
report at `gpu-lowering` before device skips or empty returns. Shaped GPU runs
support the same trailing-axis broadcast shapes as CPU within target limits. Strict Metal policy applies to every nonempty
GPU execution. Empty inferred shapes, including broadcasts with nonempty inputs, succeed
without a device and emit no MSL.
`shaped_gpu_contracts` and `shaped_gpu_stub_contracts` check extraction/export,
metadata lifetime/corruption, malformed broadcast index paths, empty/unavailable
behavior, unequal-buffer overlap and direct ownership.
`shaped_gpu_reference_fixtures` uses the independent nested-loop oracle and
checks exact generated contents read-only. Regenerate with
`python3 test/reference/generate_shaped_gpu.py`, or add `--check` to verify.
`gpu_broadcast_reference_fixtures` adds four grouped GPU broadcast files checked
against the independent nested-loop oracle. Regenerate with
`python3 test/reference/generate_gpu_broadcast.py` or verify with `--check`.
The two direct GPU contract executables now take both
`test/shaped_gpu/semantics.ml` and `test/broadcast_gpu/semantics.ml` as arguments.
See [usage](shaped-tensor-metal.md), [M3-C evidence](m3-c-acceptance.md) and
[M3-D evidence](m3-d-acceptance.md).

## Full M3 acceptance

`test/acceptance_m3/` closes the shaped milestone with 30 named cases / 64 runs
in three files: seeded semantics, exact flattened grid boundaries and empty
broadcasts. Grid widths 4/32/256 each exercise counts `1`, `W-1`, `W`, `W+1` and
`2W+3`, while shape/arity choices are distributed rather than cross-multiplied.
Inputs include MIN/MAX/-1/0/1 and independent full-width LCG streams with seed
`0x5c0e0003`. An independent Python nested-loop oracle computes shape and
wrapping i32 values without using production indexing or compiler output.

Two complete, hand-authored MSL files in `test/reference/msl/` are embedded in
the source cases as `text-equal` expectations. They cover every binding, scalar
SSA value, guard and identity/row/column/singleton index, including unsigned
semantics. They are never refreshed from backend output. `device` runs also
check GPU IR and source preservation without requiring Metal.

```sh
python3 test/reference/generate_m3.py --check
./build/tester --suite test/acceptance_m3 --require-metal
./build/tester --file test/acceptance_m3/grid-boundaries.ml --case w256-n257 --run metal --require-metal
```

`shaped_acceptance_reference_fixtures` verifies oracle/data anchors and exact
fixture contents/file set read-only. Regenerate fixtures with the same Python
script without `--check`; intentional shader changes require explicit review
and edits to the hand-authored MSL expectations. Existing shaped/flat API,
malformed IR, runtime, stub and reference regressions remain required.
See [M3 acceptance](m3-acceptance.md) for the complete requirement/evidence map,
commands, actual device results and limits.

## IR format

IR cases use native `//` comments and terminate each metadata block with `// @end`. For example:

```mlir
// @test
// version: 1
// input: mlir
// runs:
//   - name: rejected
//     pipeline: tensor-msl
// @end
// @assert:rejected:error
// stage: msl-export
// op: builtin.module
// message: exactly one gpu.module
// @end
module {}
```

An `ir-equal` or `text-equal` block contains the complete expected output, using the same comment convention. No sibling expected file is needed. See [shared_square.mlir](../test/kernel_lowering/shared_square.mlir) for source IR, complete expected GPU IR, SSA capture checks, and complete expected MSL in one file. [Exporter negative cases](../test/metal_exporter/errors) write the actual invalid input directly rather than modifying strings at runtime.

## Pipelines

| Pipeline | Behavior and available outputs |
| --- | --- |
| `verify` | Parse IR without automatic verification, then explicitly verify; success/error |
| `letalg` | Source translation or raw IR verification; initial IR/text |
| `letalg-opt` | Unwrap let, convert closures, canonicalize and CSE; optimized IR/text |
| `tensor-effects` | Verify and inspect map MemoryEffectOpInterface; structured effects text |
| `tensor-scf` | Production map-to-SCF pass, with automatic pass verification disabled so its own input checks are exercised; lowered IR |
| `tensor-gpu-ir` | Extract exactly one map using the production GPU lowering; device IR and source-preservation result |
| `tensor-msl` | Lower a source/LetAlg map if present, then export the GPU module; GPU IR, MSL and preservation result |
| `metal-compile` | Same export path, then compile MSL and create the entrypoint's pipeline; no dispatch |
| `tensor-cpu` | Production CPU tensor runner; typed tensor result and lowered LLVM IR |
| `scalar-cpu` | Scalar source lowering and CPU JIT; typed scalar result |
| `tensor-metal` | Production GPU runner: frontend, GPU IR extraction, MSL export and synchronous Metal dispatch; tensor result and actual executed MSL |

Legacy `@letalg:opt` and `@runner:cpu` blocks continue to run unchanged. Their implicit `legacy-letalg`/`legacy-cpu` adapters preserve the previous scope/closure pipeline, function representation, exact-text matching, and scalar results. They do not silently add canonicalization. Legacy and version-1 metadata cannot be mixed in one case; named cases require version-1 metadata.

`tensor-metal` replaces the former `tensor-metal-legacy` adapter after runtime integration. Existing tensor execution cases now exercise the new exporter path. For dispatch boundaries see `test/tensor/dispatch`: a width of 4 is configured alongside the program and expected result in each file.

## Executing raw map IR

`tensor-cpu-ir` and `tensor-metal-ir` execute a test-only static rank-1 i32 map
with explicit signed host data. They let tests cover negative inputs, extreme
constants and shared SSA without changing the source grammar or public runner
API. The file-level `tensor-input` YAML array is required for these pipelines
and must exactly match the static shape. Empty arrays are supported. The field
is rejected on source cases or when no IR execution run consumes it.

The module must contain exactly one function with two buffer arguments, no
results, and a body containing only the map followed by `func.return`. Layout
must be identity in the default host memory space. Wrapper/length errors are
reported as `runner`, so they cannot satisfy expected compiler-error assertions.

| Pipeline | Execution and assertions |
| --- | --- |
| `tensor-cpu-ir` | Production SCF lowering followed by standard LLVM lowering/JIT; result, compare, lowered IR and success/error |
| `tensor-metal-ir` | Production GPU lowering, exporter and runtime; result, compare, GPU IR, MSL, entrypoint, preservation and success/error |

See [031-shared.mlir](../test/acceptance/ir/031-shared.mlir) for a complete
example with host input, CPU execution, explicit/default-width Metal execution,
full expected arrays and SSA sharing checks. CPU calls use guarded host buffers
and check that input and output canaries are preserved. Numeric failures report
the first differing element. GPU execution uses the normal runtime's exact-size
Metal buffers; the host canaries are not a claim of GPU memory instrumentation.

Each Metal IR run lowers and exports before deciding device availability.
Nonempty runs skip if Metal is unavailable or disabled; strict mode rejects
those skips. Empty runs invoke the runtime no-op and pass without a device,
including under `--no-metal`, because no compilation or dispatch is needed.

[generate.py](../test/reference/generate.py) independently computes expected
values with masked Python integers. It never executes the compiler or reads
actual backend results. Run `python3 test/reference/generate.py --check` to
verify the committed fixture set, or omit `--check` to deliberately regenerate
it. Ordinary `tester` reads these self-contained files and never runs scripts.
See the [acceptance record](m1-acceptance.md) for the seed, matrix and results.

## Production external-input source runs

M2-A adds `tensor-input-cpu` (the production `runTensorCPU` API) and
`tensor-input-ir` (shared frontend only). All external-input pipelines require source input and a
per-run `arguments` list of `{name, values}`. See
[external input usage](external-tensor-input.md) for a full data-file example.
The array values are i32; unknown fields and values outside i32 fail schema
validation. An explicit empty argument list reaches production arity validation,
while an omitted field is a configuration error. The field is rejected on
unrelated pipelines. `tensor-input-metal` is now registered for M2-B and accepts positive
`threads-per-group`. Unimplemented pipeline names remain errors.

`tensor-input-cpu` supports numeric result/compare, LLVM IR and success/error
assertions and checks that caller data remains unchanged. `tensor-input-ir`
supports IR/text and success/error assertions. Expression/argument errors are
reported at `frontend`. The F5 raw-IR adapters retain their separate file-level
`tensor-input` field and restricted wrapper; their behavior is unchanged.

`tensor-input-metal` calls the production `runTensorGPU` API and supports
numeric result/compare, actual MSL text and success/error assertions. Frontend
and host launch errors are checked before a device skip. Nonempty device runs
skip under `--no-metal` and fail under `--require-metal`; empty runs still export
and execute the runtime no-op without a device. Runner contracts check these
policies explicitly, including invalid expressions and widths with empty input.
`gpu_input_contracts` and `gpu_input_stub_contracts` test direct API ownership,
lifetimes, source dumps and nonempty no-device rejection; the stub is built
explicitly on macOS as well.

CTest also runs `cpu_input_contracts` for direct API ownership, lifetime,
empty-program/dump and synthetic-size checks that cannot be expressed as a
normal nonempty test program. It reads numeric programs/data from the same
fixtures and does not keep a second C++ table of compiler examples.

## Assertions

| Kind | Meaning |
| --- | --- |
| `result` | An i32 scalar or i32 array, parsed and compared by value and kind; `4` is not `[4]` |
| `compare` | Equal numeric result to an earlier numeric run in the same case |
| `ir-equal` | Parse and verify expected and actual IR, compare operations/types/attributes and SSA correspondence, ignoring locations and printed names |
| `ir-check` | LLVM FileCheck directives applied to emitted IR |
| `text-equal` | Full output equality after outer whitespace trimming, retaining the old exact-text convention |
| `text-check` | LLVM FileCheck directives applied to MSL or another supported textual output |
| `effects` | Exact structured effect output, including operand positions and unexpected/unbound effects |
| `entrypoint` | Exact exported entrypoint name, independently checked even when device compilation is unavailable |
| `source-unchanged` | `true`; GPU extraction and/or export must preserve their input IR |
| `success` | `true`; the chosen pipeline must actually complete |
| `error` | YAML with required stage/message and optional op; only a failure at that stage with matching diagnostic fragments passes |

An error assertion cannot be combined with successful-output assertions. Parser errors, pipeline-input selection errors, configuration errors, and assertion errors cannot satisfy an expected failure at a later compiler stage. Internal adapter-input errors are reported as `runner`, which is not a selectable expected compiler-error stage.

Supported expected stages are `parse`, `frontend`, `verify`, `optimize`, `cpu-lowering`, `gpu-lowering`, `msl-export`, `metal-compile`, `cpu-execute`, and `metal-execute`. Raw IR verification is deferred so a verifier-negative case reaches the intended stage. Source translation includes the production frontend's own verification.

FileCheck is linked as an LLVM library; the runner does not invoke shell commands or require a separate FileCheck executable. Symbolic captures can check SSA sharing:

```text
CHECK: %[[ELEMENT:.*]] = memref.load
CHECK-NEXT: %[[SQUARE:.*]] = arith.muli %[[ELEMENT]], %[[ELEMENT]] : i32
CHECK-NEXT: %[[TWICE:.*]] = arith.addi %[[SQUARE]], %[[SQUARE]]
```

Keep full structural/golden assertions where appropriate rather than replacing them with keyword checks. The shared-square and integer-bit cases retain complete expected outputs; result cases keep explicit expected arrays alongside CPU/Metal comparisons.

## Migration and boundaries

The former tensor semantics, GPU lowering, exporter, Metal compilation, and tensor target executables have been replaced by data cases and the unified adapters. Their obsolete source/expected pairs and C++ case tables were removed; Git history retains them. The standalone frontend/extraction and malformed-memory-IR contracts remain in `Interfaces.cpp` because they specifically test C++ linkage, automatic dialect loading, and object mutation, not text programs.

CI and local CTest use the same runner. `RunnerChecks.cmake` generates temporary malformed configurations under the build directory and requires both a failing exit status and the expected reason. This prevents configuration typos, wrong error stages, and missing selections from appearing green.

On the installed LLVM 22.1 distribution, ExecutionEngine depends on the monolithic MLIR/LLVM libraries. The unified targets use that linkage consistently to avoid duplicate MLIR interface TypeIDs from mixing component archives with the shared library. LLVM FileCheck remains a separate archive, resolving its support symbols through LLVM.

The test migration preserves compiler semantics. Runtime integration additionally provides `tensor-metal` execution through the GPU IR/exporter path without adding language syntax, file imports, or dynamic tensor inputs.

## Multi-input source execution (M2-C)

Existing `tensor-input-cpu`, `tensor-input-ir` and `tensor-input-metal` runs now
accept 1-3 declared arguments. No pipeline names or source grammar changed.
Arguments must have unique names and equal lengths; even unused arguments are
validated. `test/multi_input` contains explicit CPU/GPU expected arrays, binding
order, repeated/unused input, signed wrapping, empty/mismatched shape and arity
cases. The former two-argument rejection case is now a positive unused-input
case; zero/four arguments remain errors.

Raw IR cases check variadic body arity, per-operand read effects (including
repeated SSA inputs), output alias rejection, layout/capture/effect restrictions,
ordered GPU loads, shared SSA and complete three-input MSL. F5's raw execution
adapter intentionally retains its single-input wrapper precondition. Direct
CPU/GPU input contracts additionally read `multi_input/alias.ml` to exercise two
host spans pointing to the same storage; ordinary numeric fixtures alone cannot
prove actual host aliasing. Runtime contracts check exported arity mismatch,
all-input capacity checks and valid empty two/three-input calls on real/stub APIs.

## M2 independent acceptance matrix

`test/acceptance_m2` contains 70 self-contained production-source cases (207
runs) in seven physical files generated by `test/reference/generate_m2.py`.
The matrix files `matrix/single-input-launch-boundaries.ml`, `matrix/two-input-launch-boundaries.ml`, and `matrix/three-input-launch-boundaries.ml` each contain
22 named cases. The four ownership/semantics fixtures remain standalone for
direct API consumers. Each case contains named
host arrays, its expression, explicit/default launch settings, independent
expected arrays and CPU/Metal comparisons. The runner reads these directly;
Python is needed for fixture reproducibility, not ordinary execution.

```sh
python3 test/reference/generate_m2.py --check
python3 -O test/reference/generate_m2.py --check
./build/tester --suite test/acceptance_m2 --require-metal
./build/tester --suite test/acceptance_m2 --no-metal
```

The generator evaluates a separate instruction list with masked Python integers.
It shares the F5 signed-bit conversion, deterministic data stream and annotation
printer, but never calls the production parser or compiler. `--check` checks the
exact file set and contents without rewriting fixtures or creating bytecode
caches. CTest's `multi_input_reference_fixtures` runs this check, alongside F5's
existing `reference_fixtures` check.

`InputOwnership.h` shares direct CPU/GPU contracts using the generated lifetime
and overlap files. Actual host pointers exercise partial overlap of three inputs;
guarded input storage, changed-data repeated calls, output independence and
post-call lifetime are checked without duplicating numerical programs in C++.
The host canaries do not instrument Metal device buffer overruns. The explicit
stub checks nonempty rejection and empty success without claiming GPU execution.
Runner contracts deliberately corrupt a generated result and require failure,
check comparison dependencies and strict device policy, and require frontend
errors to pass without device skips. See [M2 acceptance](m2-acceptance.md).

## Typed f32 CPU fixtures

`shaped-f32-cpu` and `shaped-f32-ir` require `profile: f32-ftz`, and every
argument explicitly declares `dtype: f32`. Result arrays use exact binary32 bits
except NaN classification; zero and infinity signs remain significant. Decimal
and raw-bit data, schema restrictions and the numerical contract are documented
in [f32 execution](f32-execution.md). `test/f32_cpu/` contains 28 source runs;
`f32_cpu_contracts` covers ABI/ownership, malformed normalization/profile IR,
rounding environment and schema assertions. The independent native CPU/Metal
probe is `test/reference/probe_f32.mm`; [M4-A evidence](m4-a-acceptance.md)
records the decision gate and commands.


## Typed f32 Metal fixtures

`shaped-f32-metal` executes the typed production GPU API; `shaped-f32-gpu-ir`
and `shaped-f32-msl` verify device extraction/export and preservation. They use
the same explicit dtype/profile and exact result policy as f32 CPU. The 45 runs
in `test/f32_gpu/` cover actual arithmetic and structural/error contracts;
`f32_gpu_contracts` / `f32_gpu_stub_contracts` add typed runtime metadata,
corruption, ownership and unavailable/empty behavior. The stub target builds on
macOS as well. See [M4-B evidence](m4-b-acceptance.md).

## Complete M4 numerical acceptance

`test/acceptance_m4/` contains 62 self-contained cases / 125 runs across exact
arithmetic, exceptional-value cross products, broadcasting, dispatch boundaries,
empty outputs, seeded wide magnitudes and a complete hand-authored f32 shader.
`f32_acceptance_reference_fixtures` checks the independent integer/Fraction
binary32 oracle's anchors, exact contents and file set read-only. Every numeric
run checks independent expectations and shape; Metal also compares with CPU.

```sh
python3 test/reference/generate_m4.py --check
python3 -O test/reference/generate_m4.py --check
./build/tester --suite test/acceptance_m4 --require-metal
./build/tester --suite test/acceptance_m4 --no-metal
```

Runner contracts mutate exact results, zero signs, profile/dtype, missing
shape/values, comparisons and the shader golden, and verify strict/empty/error
device policy. See [M4 acceptance](m4-acceptance.md) for the oracle design,
coverage/evidence map, numerical contract and final results.

## Sum CPU execution

`sum-i32-cpu` / `sum-f32-cpu` execute the production fixed-tree sum APIs;
`sum-i32-ir` / `sum-f32-ir` expose explicit Linalg identities/reduction iterators.
Sum execution requires both values and shape assertions, and uniquely accepts
rank-0 shape `[]`. F32 runs require `profile: f32-sum-tree256-ftz`.
`sum_cpu_contracts` verifies ABI/ownership/plan contracts; `sum_reference_fixtures`
checks the independent wrapping/rational tree oracle without writing files.
See [sum usage](sum-execution.md) and [M5-A evidence](m5-a-acceptance.md).

`sum-i32-metal` / `sum-f32-metal` execute the real/stub fixed-tree runtime;
`sum-i32-msl` / `sum-f32-msl` expose its canonical shader. The 61 runs in
`test/sum_gpu/` cover actual arithmetic, empty/device policy and structural
barrier/indexing checks. `sum_gpu_contracts` / `sum_gpu_stub_contracts` verify
metadata/source/dispatch corruption, byte limits and direct ownership.
See [M5-B evidence](m5-b-acceptance.md).

`test/map_sum/semantics.ml` adds 16 cases / 32 CPU/Metal runs for materialized
broadcast-expression reductions, including the `[140,320]` row-sum example.
The existing direct sum executables also consume this file to test actual
borrowed-span overlap and output lifetime. See [M5-C evidence](m5-c-acceptance.md).

## Full M5 acceptance and walkthrough

`test/acceptance_m5/` adds 47 cases / 92 runs covering actual three-dispatch
65,537-item execution, strided axes, numerical bounds, exceptional cross-group
arithmetic, composed broadcasts, invalid source and complete hand-authored i32/f32
MSL. The sum generator now checks 10 files, retaining all A/B/C fixtures.
Runner contracts prove wrong sum bits/shapes/profile/zero/goldens fail, invalid
launches precede skips, and empty reductions succeed in strict no-device mode.
`sum_walkthrough` is a CPU CTest and can be run manually with `--metal`, `--ir`,
`--msl`; see [full evidence](m5-acceptance.md).
