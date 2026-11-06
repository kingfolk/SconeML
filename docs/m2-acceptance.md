# M2 acceptance: external and multi-input CPU/Metal expressions

M2-A through M2-D are implemented and accepted locally on top of `88e284e`
(M2-C, PR #17). This record covers M2-D's test changes and local evidence.
It does not claim that the M2-D change set is merged or validated on other GPUs.

## Supported production behavior

`runTensorCPU(expression, arguments)` and `runTensorGPU(expression, arguments)`
accept 1–3 uniquely named, equal-length, contiguous rank-1 i32 host spans.
Expressions use variable names, nonnegative i32 constants and wrapping
addition/subtraction/multiplication. Negative and full-width signed values are
supplied through host arrays. Each call may translate, compile and execute anew.
See [usage and ABI](multi-input-tensor-map.md) for a complete C++ example.

Every argument is validated even when unused. Declaration order controls ABI
binding, while expression names select scalar operands at the current index.
CPU dispatch uses the exact K+1 descriptor signature; Metal binds K ordered
input buffers, an independent output and count. Repeated and partially
overlapping read-only host input spans are allowed. Returned vectors own their
storage and survive destruction of inputs and per-call JIT/device resources.

Inputs must remain alive for the synchronous call. All-empty calls still reject
invalid names, arity, expression syntax and lengths. Valid empty CPU calls skip
JIT and clear the LLVM dump; valid empty GPU calls export MSL and skip device
compilation/dispatch. Explicit nonempty GPU requests fail without a device;
there is no CPU fallback.

Shapes are specialized for each call; buffers contain runtime data, not embedded
constants. Broadcasting, rank-2 inputs, f32, reductions, imports, persistent
sessions, compilation caches and asynchronous execution remain outside M2.

## Independent oracle and data matrix

`test/reference/generate_m2.py` uses Python integer instructions, masks every
result to 32 bits and interprets the final bit pattern as signed i32. It never
imports production parsing/lowering, runs SconeML, or consumes backend output.
Hand-calculated anchors check each matrix expression, MAX+1, MIN-1,
multiplication wraparound, zero length and mismatched-length rejection.

Seed: `0x5c0e0002`. Input arrays use distinct deterministic streams and displaced
MIN/MAX/-1/zero/one/multiplication-boundary anchors. The shared F5 xorshift32
helper provides full-width values beyond the anchors. Declaration order is
reversed relative to the a/b/c names. Three matrix expressions cover precedence,
left associativity, repeated operands and all supported arithmetic operations.

For every K=1/2/3, generate the following explicit-width matrix. Each file also
runs CPU and an automatic-width Metal call, with explicit independent expected
arrays plus CPU/Metal comparisons. Automatic width uses the actual runtime
policy; explicit widths establish known boundary geometry independently.

| Explicit width W | Lengths |
| ---: | --- |
| 1 | 0, 1, 2, 3 |
| 4 | 0, 1, 3, 4, 5, 9 |
| 32 | 0, 1, 31, 32, 33, 65 |
| 256 | 0, 1, 255, 256, 257, 513 |

These 66 files cover 0/1/W-1/W/W+1/2W+1 (duplicates removed), exact groups,
padded groups and multi-group tails. Four additional files cover unused inputs,
a constant expression, changed-data lifetime calls and partially overlapping
input values. Total: 70 files, 207 runs.

Existing `test/multi_input/errors` and IR fixtures remain active: duplicate/
invalid/unknown names, zero/four arguments, mismatched/unused/empty lengths,
variadic region arity, layout/type/shape, effects, aliases and ordered exporter
loads. Invalid input cases are asserted errors, never device skips.

## Direct API and runner contracts

The CPU and GPU direct contracts read the same lifetime/overlap data files.
They check guarded input storage, output independence, changed-data repeated
calls, persistence of earlier results and results surviving input destruction.
Three actual input spans overlap at different offsets in one host allocation;
the older two-name same-address check also remains. Ordinary YAML arrays alone
would not prove either pointer relationship. Metal copies these spans into
independent device input buffers. Host canaries are not device memory guards.

The real GPU contract additionally requests an excessive but u32-representable
width and requires a supported-range error. Synthetic runtime checks cover
capacity/bytes/arity/padded-index bounds and pipeline-limit planning. Empty
three-input success and nonempty stub rejection are checked directly.

Runner contracts deliberately corrupt a generated expected value and require
an element mismatch. They also check automatic inclusion of CPU comparison
dependencies, required-device failure, empty success without a device, and all
multi-input frontend errors under strict no-device mode. Fixture reproducibility
runs under CTest and the existing PR CI workflow.

## Local validation

Environment: Apple M1, arm64, macOS 15.5, LLVM/MLIR 22.1.8,
Python 3.14.6. Compiler: `/opt/homebrew/opt/llvm@22/bin/clang++`.
The sandbox does not expose the Metal device; real-device checks below ran
outside that sandbox through the tool permission system.

```sh
cmake -S . -B build
cmake --build build --target tester tensor_interfaces metal_runtime_checks metal_stub_checks cpu_input_contracts gpu_input_contracts gpu_input_stub_contracts -j3
python3 test/reference/generate_m2.py --check
python3 -O test/reference/generate_m2.py --check
python3 test/reference/generate.py --check
ctest --test-dir build --output-on-failure
./build/tester --suite test/acceptance_m2 --require-metal
./build/tester --require-metal
./build/tester --no-metal
./build/gpu_input_contracts test/external_input/polynomial.ml test/multi_input/alias.ml \
  test/acceptance_m2/lifetime.ml test/acceptance_m2/overlap.ml
```

| Check | Observed result |
| --- | --- |
| CTest including references, runner, API, runtime and explicit stub contracts | 10/10 passed |
| M2-D generated matrix, Metal required | 207 passed, 0 skipped, 0 failed |
| M2-D matrix, Metal disabled | 95 passed, 112 skipped, 0 failed |
| Complete data suite, Metal required | 536 passed, 0 skipped, 0 failed |
| Complete data suite, Metal disabled | 339 passed, 197 skipped, 0 failed |
| Direct real-device GPU contracts | Passed on Apple M1, including three-input lifetime/overlap and excessive-width rejection |
| Independent fixture checks | 70 M2 files and 17 M1 files reproduced; M2 anchors/check also passed under optimized Python |

No-device skips are nonempty device runs; CPU/frontend/IR checks and valid
empty no-ops execute. The explicit non-Apple stub is built and tested on this
Mac, which is not separate Linux platform validation. Existing M1 fixtures and
full structural IR/MSL expectations remain in the complete suite.

This finite matrix establishes prototype correctness within the stated scope;
it does not establish performance or exhaustive memory safety. M2 is accepted
locally. The next planned capability is rank-2 shapes and broadcasting.
