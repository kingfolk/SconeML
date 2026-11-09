# M3-C acceptance: same-shape GPU and shaped launch metadata

Date: November 9, 2026 (America/New_York). Baseline: `2d61665` (M3-B, PR #21).
This records M3-C. GPU broadcasting remains deferred to the next feature.

## Delivered behavior

`runShapedTensorGPU` evaluates identical rank-1/rank-2 dense row-major i32
inputs through the shared shaped frontend, verified Linalg-to-flat TensorMap
extraction, existing device IR/exporter and actual Metal execution. Results
own both values and shape. The same-shape `[2,3]` three-input expression
`a * b + c - a` returns `[109,238,387,556,745,954]`, matching independent
expectations and CPU execution.

The extraction validates the full shaped wrapper/scalar envelope and identical
shapes before cloning scalar IR. Singleton axes with equal input/output extent
one are semantically identity. Broadcast reuse, including empty broadcast
shapes, fails explicitly before device skips. Empty identical shapes return
without extracting/exporting a kernel, allocating buffers or accessing Metal;
the optional MSL dump is empty.

Device IR contains shape, per-input counts, output count and a shaped kernel
marker. Export validates metadata against arity, shape product/strides and u32
bounds before creating owned plain launch metadata. Runtime requires exact input
lengths and expected output count, checks each buffer's host/device byte limit,
allocates/copies each input at its own count, and retains uniform padded launch,
command error propagation and readback after success. Real/stub checks share
host preflight. Old flat wrappers retain their prefix/capacity contracts.

## Independent fixtures and contracts

`generate_shaped_gpu.py` uses the independent set-based shape and nested-loop
wrapping i32 oracle from `generate_broadcast.py`. It imports no production
compiler, parser, shape helper or runtime. Source expressions and scalar
instruction lists are specified separately. Hand-checked oracle anchors are
validated before generating or checking exact fixture contents/file sets.

| File under `test/shaped_gpu/` | Cases | Runs | Purpose |
| --- | ---: | ---: | --- |
| `semantics.ml` | 16 | 34 | K=1/2/3, rank 1/2, singleton axes, signed wrapping, unused/constant expressions, ownership fixture and empty shapes; device IR/MSL checks |
| `launch-boundaries.ml` | 10 | 20 | Nonuniform signed input data, two-row matrices and explicit widths 1/4/32 across padded boundaries |
| `invalid.ml` | 9 | 9 | Reject broadcasting/rank changes, invalid empty syntax/extents and invalid launch width before device access |

Each successful Metal run independently asserts its values and exact output
shape, and compares with CPU. The device IR/MSL runs verify shape/count metadata,
input/output/count parameter order, guard/index operations, loads/stores and
source preservation without needing a device.

Direct `shaped_gpu_contracts` and explicit `shaped_gpu_stub_contracts` verify:

- Host/device module preservation and equality of shaped/legacy MSL.
- Owned metadata/source surviving MLIR context destruction and a later launch.
- Missing root/kernel markers; malformed metadata types, shapes, counts, arity,
  unknown keys and canonical-stride overflow rejected without modifying IR.
- Metadata-only u32 target rejection without huge buffers or forged spans.
- Empty calls clearing MSL, syntax validation, empty broadcasting rejection,
  and explicit nonempty unavailable/stub failure.
- Repeated calls, escaping results, independent output data/shape, aliased and
  overlapping inputs with canaries, and readback into a larger output span.
- Retained legacy flat requested-prefix behavior with a larger input span.

Shared runtime contracts add exact count/length/arity errors, distinct per-input
byte counts, independent input/output device byte limits, synthetic u32 limits,
nonempty-output/empty-input rejection, and empty output with nonempty input
metadata on both real and stub runtimes. Runner contracts check strict Metal
failure, strict empty success and all nine errors before device selection.

## Validation

Apple M1 (8-core GPU), arm64, macOS 15.5, Homebrew LLVM/MLIR and clang 22.1.8.
Real Metal validation ran outside the desktop sandbox with device access.

```sh
cmake --build build --target tester tensor_interfaces metal_runtime_checks metal_stub_checks cpu_input_contracts gpu_input_contracts gpu_input_stub_contracts grouped_spec_checks shaped_cpu_contracts shaped_gpu_contracts shaped_gpu_stub_contracts -j3
python3 test/reference/generate_shaped_gpu.py --check
./build/tester --suite test/shaped_gpu --require-metal
./build/shaped_gpu_contracts test/shaped_gpu/semantics.ml
./build/shaped_gpu_stub_contracts test/shaped_gpu/semantics.ml
ctest --test-dir build --output-on-failure
./build/tester --require-metal
./build/tester --no-metal
```

| Check | Result |
| --- | --- |
| New source subset, actual Metal required | 63 passed, 0 skipped, 0 failed; 3 files / 35 cases |
| New source subset, Metal disabled | 40 passed, 23 expected device skips, 0 failed |
| Direct shaped GPU contracts | Passed on Apple M1 |
| Explicit shaped GPU stub contracts | Passed |
| CTest, including runtime/API/runner contracts and four fixture checks | 16/16 passed |
| Complete strict Metal suite | 677 passed, 0 skipped, 0 failed; 107 files / 332 cases |
| Complete Metal-disabled suite | 457 passed, 220 expected device skips, 0 failed |

The preceding baseline had 104 files, 297 cases and 614 runs. All remain active;
this feature adds three files, 35 cases and 63 runs. Existing shader goldens and
negative IR tests pass unchanged. No performance improvement is claimed.

## Limits

Shaped GPU calls require identical shapes. CPU broadcasting remains available,
but GPU broadcast indexing and unequal-shape source execution are not delivered
here. Rank >2, arbitrary strides/views, floats, reductions, persistent sessions
and compilation caches remain outside scope. The plain runtime metadata is
paired with its validated kernel export, not an independent arbitrary-source
indexing proof. Local planning files are excluded from this feature PR.
