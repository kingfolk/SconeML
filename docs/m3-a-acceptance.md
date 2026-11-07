# M3-A acceptance: shaped same-shape CPU maps

Date: November 7, 2026. Baseline: `e7e16f0` (grouped source tests, PR #19).
This records M3-A only. Broadcasting and shaped GPU execution remain deferred.

## Delivered behavior

The production `runShapedTensorCPU` API executes one expression over 1–3 named
rank-1/rank-2 dense row-major i32 inputs with equal shapes. It returns owned data
and shape. The frontend builds flat ABI buffers, canonical reinterpret views
and a verified buffer-based `linalg.generic`. The runner lowers it through SCF
to LLVM and executes native CPU JIT code. Existing flat and literal paths retain
their contracts.

## Evidence and independent expectations

`test/shaped_input/same_shape.ml` contains 14 named cases and 15 runs; the extra
run checks frontend Linalg structure. `invalid.ml` contains 21 named rejection
cases/runs. Together they add 35 logical cases and 36 executions in two files.

Expected values are explicit data, independent of compiler/runtime output. For
example, the two-input `[2,3]` sum expects `[11,22,33,44,55,66]`; the three-input
`a * b + c - a` fixture independently expects `[109,238,387,556,745,954]`.
The wrapping fixture checks signed extrema and `65536 * 65536 + 1 == 1`
modulo 2^32. Every successful CPU fixture independently asserts its exact shape,
including `[0]`, `[0,3]` and `[2,0]`.

| Contract | Evidence |
| --- | --- |
| Same-shape rank-1/rank-2 arithmetic and K=1/2/3 ABI | Source fixtures plus direct public API calls |
| Constants, unused declarations, binding, wrapping arithmetic | `same_shape.ml` |
| Specialized static views/maps, real Linalg and LLVM elimination | Frontend/LLVM FileCheck assertions |
| Host values absent from IR; shape changes specialize IR | Direct calls compare LLVM dumps before/after value and shape changes |
| Inputs read-only, aliased/overlapping spans allowed | `shaped_cpu_contracts`, with independent alias/overlap fixture values |
| Output owns shape/data after inputs/context/JIT destruction | Direct lifetime, mutation and later-call checks |
| Rank, negative/dynamic extents, shape/storage mismatch, name/arity errors | 21 grouped negative cases |
| Index/byte/capacity/stride overflow before allocation | Metadata-only synthetic bounds checks and overflow source fixtures |
| Empty validates source and arguments, preserves shape, clears dump, skips JIT | Source cases and direct empty API contract |
| Unsupported indexing/output reads/captures/scalar ops/overflow flags rejected | Mutated in-memory wrapper/Linalg API contracts |
| Missing shape/shape on flat pipelines/incorrect result shape rejected | Runner schema and assertion contracts |
| Old flat unequal lengths still rejected | Direct compatibility contract and retained M1/M2 regressions |

## Environment and commands

Apple M1 (8-core GPU), arm64, macOS 15.5 (24F74), Homebrew LLVM/MLIR and clang
22.1.8, existing CMake build. Real Metal execution requires the device access
provided outside the desktop sandbox.

```sh
cmake --build build --target tester tensor_interfaces metal_runtime_checks metal_stub_checks cpu_input_contracts gpu_input_contracts gpu_input_stub_contracts grouped_spec_checks shaped_cpu_contracts -j3
./build/tester --suite test/shaped_input --no-metal
./build/shaped_cpu_contracts test/shaped_input/same_shape.ml
ctest --test-dir build --output-on-failure
./build/tester --require-metal
./build/tester --no-metal
```

| Check | Result |
| --- | --- |
| Shaped source subset | 36 passed, 0 skipped, 0 failed; 2 files / 35 cases |
| Direct shaped CPU contracts | Passed |
| CTest (including explicit stub, API, runner and reference fixture checks) | 12/12 passed |
| Complete suite, actual Metal required | 572 passed, 0 skipped, 0 failed; 101 files / 256 cases |
| Complete suite, Metal disabled | 375 passed, 197 expected device skips, 0 failed |
| Prior M1/M2 independent reference checks | Passed unchanged through CTest |

## Limits

All inputs must have identical shapes; singleton and trailing-axis broadcasting
fail explicitly. Shape rank is limited to 1 or 2, storage is contiguous, and
arithmetic remains homogeneous wrapping i32. There is no shaped GPU API,
source-syntax extension, general view support, reduction, cache or session.
This is an execution/contract proof, not a performance claim or full M3
acceptance. Local planning documents remain excluded from the feature PR.
