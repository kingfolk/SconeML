# M1 acceptance: CPU and Metal tensor maps

M1 implementation and acceptance are complete locally on top of `d0fe100`
(F4, PR #13). This record covers the F5 change set and its local validation;
it does not claim validation on other GPUs. Merge status is tracked separately
from the acceptance results.

## Scope and evidence

The supported source path remains one static rank-1 i32 tensor literal, one
map and an independent output, with wrapping constants/add/subtract/multiply.
Both source runners use the shared frontend; the GPU runner uses explicit GPU
IR, the independent Metal exporter and synchronous uniform dispatch.

F5 adds 17 self-contained data files under `test/acceptance`, containing 46
named runs. Existing source, verifier, effects, lowering, exporter and runtime
checks remain active. No production source-language or execution API changed.

| Entry point | What the tests establish |
| --- | --- |
| Source programs | CPU and actual GPU results match independent expected arrays for add, subtraction/negation, square and a polynomial with overflow; empty source literals remain rejected at the frontend |
| Raw LetAlg map | Full-width signed i32 inputs, extreme constants, shared SSA and wrapping arithmetic execute through production SCF lowering or GPU lowering/export/runtime |
| Raw GPU/export fixtures | Existing complete IR/MSL expectations, unsigned indexing/guard validation, preservation, target compilation and unsupported-operation diagnostics remain passing |
| Runtime API | Zero count does not access a device or dispatch; synthetic count/capacity/byte/padded-index limits, error propagation, and the non-Apple stub remain checked |

Raw IR execution is a test adapter, not support for external inputs in the
source API. It accepts only a single static identity-layout map wrapper and
reads its signed input array from the same `.mlir` file as the program and
assertions. CPU execution uses guarded host buffers, verifies input preservation
and checks output canaries. GPU runs use the production exact-size allocations
and generated bounds guard; this is not a GPU sanitizer or exhaustive proof of
memory safety.

## Independent reference and reproducibility

`test/reference/generate.py` has no compiler/MLIR imports and does not execute a
backend or consume its outputs. Its oracle evaluates an explicit arithmetic
DAG using Python integers, masks every constant and result with `0xffffffff`,
and interprets the final bit pattern as signed i32. This implements uint32
modular arithmetic without C++ signed overflow. Hand-calculated anchors check
MAX+1, MIN-1, multiplication wraparound and shared-square behavior before any
fixture generation or comparison. The checks also run under optimized Python.

The seed is `0x5c0e2026`. A specified xorshift32 stream supplies full-width
signed values after fixed anchors including MIN/MAX, -1, 0, 1 and multiplication
boundaries. Programs cover identity, add, subtract, multiplication, negation,
shared square, extreme constants and mixed arithmetic. The reference evaluator
and MLIR fixture printer are separate functions; neither imports production
lowering or emitter code.

```sh
# Deliberately regenerate programs/configuration/expected results together.
python3 test/reference/generate.py
# Read-only: verify anchor results, exact contents and the generated file set.
python3 test/reference/generate.py --check
```

Expected arrays are stored in each file, with both explicit `result` assertions
and CPU/Metal `compare` assertions. Ordinary test execution does not run a
script or derive expectations from actual results. Mismatches report the file,
run and first differing element. CTest and CI check reproducibility.

## Execution matrix

The 12 raw-IR cases each run on CPU, Metal with the stated explicit width, and
Metal with automatic width (`min(256, pipeline limit, device x limit)`).
Explicit width-256 cases check the boundary without relying on an assumed
automatic width. The existing F4 width-4 identity
case also covers length 1 with padding.

| Length | Explicit width | Program |
| ---: | ---: | --- |
| 0 | 4 | Identity: CPU and runtime empty no-op |
| 1 | 1 | MAX+1 |
| 3 | 4 | Subtract, including MIN-1 |
| 4 | 4 | Multiply |
| 5 | 4 | Negate |
| 31 | 32 | Shared square, one SSA square consumed twice |
| 32 | 32 | MIN/MAX/-1/zero constants and all three operations |
| 33 | 32 | Mixed arithmetic |
| 255 | 256 | Add |
| 256 | 256 | Subtract |
| 257 | 256 | Multiply |
| 513 | 32 | Mixed arithmetic across multiple groups and a partial tail |

Four additional source programs use nonnegative literals supported by the
existing parser and independently computed expected results. The fifth source
case checks empty-literal rejection on both runner paths. Negative host inputs
and negative/extreme constants are explicitly tested at the IR boundary.

## Local acceptance run

Environment: Apple M1, arm64, macOS 15.5, LLVM/MLIR 22.1.8, Python 3.14.6.
The project uses `/opt/homebrew/opt/llvm@22/bin/clang++`.

```sh
cmake -S . -B build
cmake --build build --target tester tensor_interfaces metal_runtime_checks metal_stub_checks -j3
ctest --test-dir build --output-on-failure
./build/tester --suite test/acceptance --require-metal
./build/tester --require-metal
./build/tester --no-metal
./build/metal_runtime_checks
./build/metal_stub_checks
```

| Check | Observed result |
| --- | --- |
| CTest, including fixture reproducibility and runner contracts | 6/6 passed |
| F5 matrix, Metal required | 46 passed, 0 skipped, 0 failed |
| Complete data suite, Metal required | 147 passed, 0 skipped, 0 failed |
| Complete data suite, Metal disabled | 110 passed, 37 skipped, 0 failed |
| Runtime API on Apple M1 | Launch/empty contracts and compilation/entrypoint error propagation passed |
| Explicit stub build | Launch/empty contracts, unavailable-device reporting and nonempty rejection passed |

The 37 no-device skips are explicitly reported nonempty Metal runs; CPU/IR
runs still execute. Empty IR execution passes without a GPU because runtime
count zero intentionally bypasses compilation and dispatch. The stub is built
and tested on this Mac, which checks the actual non-Apple implementation but
does not constitute a Linux toolchain/platform validation.

## Remaining limits

This is a finite correctness matrix for the restricted prototype, not a
performance claim or exhaustive fuzzing. Source imports, external tensor
arguments, dynamic shapes, multiple map inputs, floating point, reductions,
cache/resource reuse, asynchronous execution and additional GPU backends remain
outside M1. Broader capability work belongs in a separately planned milestone.
