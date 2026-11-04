# Data-driven compiler tests

`tester` reads each test's program, configuration, and assertions from one file. `.ml` files contain source programs; `.mlir` files contain raw IR. The runner calls production frontend, lowering, exporter, and execution code through a fixed set of pipeline adapters.

## Running tests

```sh
cmake -S . -B build
cmake --build build --target tester tensor_interfaces -j3
ctest --test-dir build --output-on-failure
```

`compiler_suite` runs all compiler data files. `runner_contracts` checks the runner's error handling, selection, and device policy. `tensor_interfaces` retains two C++ API checks: frontend linkage without CPURunner/ExecutionEngine, and rejection of malformed in-memory IR that the text parser cannot construct. Both API tests read their programs from the regular data files.

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

`version`, `input`, and `runs` are required YAML fields. Each run has only `name` and `pipeline`. Unknown/duplicate YAML fields, duplicate names or assertion kinds, assertions for unknown runs, empty programs, and missing assertions fail validation before execution. Assertion kinds must be supported by their pipeline. Arbitrary shell commands and pass scripts are not supported.

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
| `tensor-metal-legacy` | Existing GPU runner, explicitly distinguished from the new IR/export path; tensor result and MSL |

Legacy `@letalg:opt` and `@runner:cpu` blocks continue to run unchanged. Their implicit `legacy-letalg`/`legacy-cpu` adapters preserve the previous scope/closure pipeline, function representation, exact-text matching, and scalar results. They do not silently add canonicalization. Legacy and version-1 metadata cannot be mixed in one file.

`tensor-metal` is intentionally not registered yet: dispatch through the new kernel IR/exporter belongs to the subsequent runtime integration feature.

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

This feature reorganizes testing only. It does not add language syntax, file imports, dynamic tensor inputs, or a new Metal execution path.
