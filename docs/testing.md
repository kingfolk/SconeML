# Data-driven compiler tests

`tester` reads each test's program, configuration, and assertions from one file. `.ml` files contain source programs; `.mlir` files contain raw IR. The runner calls production frontend, lowering, exporter, and execution code through a fixed set of pipeline adapters.

## Running tests

```sh
cmake -S . -B build
cmake --build build --target tester tensor_interfaces metal_runtime_checks metal_stub_checks cpu_input_contracts gpu_input_contracts gpu_input_stub_contracts -j3
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

`--file` accepts an existing path, a unique basename, or a suite-relative path. `--run` selects an exact run name; comparison dependencies are automatically included. Missing files/runs, ambiguous basenames, empty suites, bad configuration, and failed assertions exit nonzero.

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

Legacy `@letalg:opt` and `@runner:cpu` blocks continue to run unchanged. Their implicit `legacy-letalg`/`legacy-cpu` adapters preserve the previous scope/closure pipeline, function representation, exact-text matching, and scalar results. They do not silently add canonicalization. Legacy and version-1 metadata cannot be mixed in one file.

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
| `compare` | Equal numeric result to an earlier numeric run in the same file |
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
