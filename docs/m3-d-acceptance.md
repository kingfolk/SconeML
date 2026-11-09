# M3-D acceptance: GPU broadcasting and checked indexing

Date: November 9, 2026 (America/New_York). Baseline: `900bef6` (M3-C, PR #22).
The local main branch was fast-forwarded to that merged baseline before this
feature branch was created. This records M3-D, not full M3-E acceptance.

## Delivered behavior

`runShapedTensorGPU` executes trailing-axis rank-1/rank-2 i32 broadcasts using
original per-input buffers. The matrix `[2,3]`, row `[3]` and column `[2,1]`
example evaluates `a + b + c` to `[111,122,133,214,225,236]` with shape `[2,3]`
on actual Metal. Outer, singleton, missing-leading-axis and zero-extent rules
match the CPU contract. All declared inputs participate, including unused inputs
and constant-only expressions.

Identical input shapes preserve the existing M3-C lowering, shader and metadata
path. Other compatible shapes lower directly from validated Linalg scalar IR
to a separately marked guarded GPU kernel. Input indices normalize to `i`,
`0`, `i / C` or `i % C`; output always stores at `i`. Equivalent singleton/shape
rank changes normalize to identity where input/output counts match. Each input
loads once and repeated scalar references share the loaded SSA value.

Broadcast metadata owns input shapes, output shape and per-buffer counts.
Export re-infers shape compatibility and normalized indexing, validates types,
markers, arity, products, strides and target counts, and checks exact index
operations, divisors, dependencies, load bindings and output index. Div/rem is
allowed only on that checked index path, never in scalar i32 expressions or
unmarked legacy IR. MSL remains unsigned and guards all index arithmetic/loads.

For nonempty output, C is positive and bounded by outputCount <= UINT32_MAX.
Within `i < R*C`, identity is below inputCount, singleton zero is below one,
row `i/C` is below R and column `i%C` is below C. Host launch preflight retains
the u32 padded-grid bound. Per-input count limits apply even to unused inputs
or empty-output calls. Empty output skips kernel extraction/export and device
access; even direct empty device extraction emits no div/rem.

## Independent fixtures

`generate_gpu_broadcast.py` uses the independent set-based shape inference and
nested-loop wrapping i32 oracle from `generate_broadcast.py`. Source expressions
and scalar instruction lists are separate from the oracle. Hand anchors cover
row/column/outer values, overflow and zero/singleton rules; no production
compiler/parser/runtime/output is used. The read-only `--check` validates
anchors, exact file set and all generated contents.

| File under `test/broadcast_gpu/` | Cases | Runs | Purpose |
| --- | ---: | ---: | --- |
| `semantics.ml` | 21 | 44 | Full example, six declaration orders, row/column/outer/singleton forms, rank-change identity, constants, unused inputs, wrapping and overlap; exact GPU IR/MSL checks |
| `empty.ml` | 8 | 16 | Zero axes, either input nonempty, missing axes and constant-only empty output |
| `launch-boundaries.ml` | 15 | 30 | Distinct full-width signed values, nonsquare shapes and explicit widths 1/4/32 with padded groups |
| `invalid.ml` | 6 | 6 | Incompatible axes, storage/syntax errors and invalid launch bounds before device skips |

Every successful GPU run independently asserts its values and exact shape and
compares against CPU. The structural case checks guard placement, each exact
input index/positive divisor, one load per input, scalar SSA dependencies,
output index, buffer order, metadata counts and source preservation.

Five former M3-C broadcast rejection cases now execute successfully and compare
against independent CPU/reference results. They moved from `shaped_gpu/invalid.ml`
to `semantics.ml`; the same 35 logical cases remain, increasing their runs from
63 to 68. The four remaining M3-C rejection cases retain error coverage. No
legacy shader golden, scalar rejection or same-shape regression was removed.

## Direct and negative contracts

The existing real/stub GPU contract targets also compile
`BroadcastGPUContracts.cpp` and read the new semantic fixtures. They check:

- Host/device IR preservation and owned `[6,3,2]` input / 6 output metadata after
  MLIR context destruction.
- Missing tags, malformed metadata types/arity/counts/extents/keys, and
  compatible metadata claiming a different row/column map than the actual IR.
- Wrong input bindings/load indices, zero or wrong divisors, swapped div/rem,
  wrong numerator dependencies, exact-division flags, signed guards, output
  index changes, extra offset arithmetic and index operations outside the guard.
- Scalar i32 division remains rejected, and removing broadcast tags does not
  authorize div/rem in legacy kernels.
- Empty device IR has no div/rem; the public empty API clears its dump and
  preserves the inferred shape without a device.
- Per-input u32 overflow even with empty output, and inferred output overflow,
  through metadata only without huge allocations or forged spans.
- Actual unequal-size GPU buffers, overlapping input spans/canaries, independent
  output data/shape, changed host values across calls and escaping results.
- Readback through metadata that outlives its compiler context into a larger
  output span, unchanged output canaries and rejection of a short row buffer.

Runner contracts enforce strict nonempty-device failure, all eight empty cases
succeeding with Metal disabled in strict mode, and all six invalid cases failing
at their expected stage before device selection. The existing per-buffer runtime
allocations/copies and count/length/device byte checks are reused unchanged.

## Validation

Apple M1 (8-core GPU), arm64, macOS 15.5, Homebrew LLVM/MLIR and clang 22.1.8.
Real Metal checks ran outside the desktop sandbox with device access.

```sh
cmake --build build --target tester tensor_interfaces metal_runtime_checks metal_stub_checks cpu_input_contracts gpu_input_contracts gpu_input_stub_contracts grouped_spec_checks shaped_cpu_contracts shaped_gpu_contracts shaped_gpu_stub_contracts -j3
python3 test/reference/generate_gpu_broadcast.py --check
python3 test/reference/generate_shaped_gpu.py --check
./build/tester --suite test/broadcast_gpu --require-metal
./build/shaped_gpu_contracts test/shaped_gpu/semantics.ml test/broadcast_gpu/semantics.ml
./build/shaped_gpu_stub_contracts test/shaped_gpu/semantics.ml test/broadcast_gpu/semantics.ml
ctest --test-dir build --output-on-failure
./build/tester --require-metal
./build/tester --no-metal
```

| Check | Result |
| --- | --- |
| New broadcast subset, actual Metal required | 96 passed, 0 skipped, 0 failed; 4 files / 50 cases |
| New broadcast subset, Metal disabled | 60 passed, 36 expected device skips, 0 failed |
| Direct real/stub GPU contracts, including malformed indices and ownership | Passed; actual device Apple M1 |
| CTest, including five independent fixture checks | 17/17 passed |
| Complete strict Metal suite | 778 passed, 0 skipped, 0 failed; 111 files / 382 cases |
| Complete Metal-disabled suite | 518 passed, 260 expected device skips, 0 failed |

The baseline was 107 files / 332 cases / 677 runs. This feature adds 4 files,
50 cases and 96 runs, plus 5 runs when former broadcast rejection cases become
CPU/Metal comparisons. The total increases by 101 runs. No performance claim
or full-M3 acceptance claim is made.

## Limits

Only 1–3 dense contiguous rank-1/rank-2 i32 inputs and the existing wrapping
`+ - *` grammar are supported. Arbitrary strides/views, transposes, higher rank,
floats, reductions, caching and persistent sessions remain out of scope.
Plain runtime metadata must stay paired with its verified source/entrypoint;
it is not an independent arbitrary-MSL indexing proof. Planning files remain
local and excluded from the feature PR.
