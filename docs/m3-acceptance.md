# M3 acceptance: shaped i32 execution and broadcasting

Date: November 9, 2026 (America/New_York). Baseline: `3d85480` (M3-D, PR #23).
The local main branch was fast-forwarded to the merged baseline before M3-E.
This record covers the complete M3-T/A/B/C/D/E gate; local acceptance and PR
merge status remain separate.

## Delivered source-to-result contract

`runShapedTensorCPU` and `runShapedTensorGPU` synchronously evaluate an existing
`+ - *` expression over 1–3 named dense rank-1/rank-2 i32 inputs and return owned
values and shape. Trailing-axis broadcasting handles matrix rows, columns,
outer combinations, singletons and missing leading axes. All declared inputs,
including unused inputs and constant expressions, determine the output shape.
Zero with one gives zero; incompatible zeros fail. Arithmetic wraps modulo 2^32.

The frontend validates names, ranks, extents, checked products/storage, canonical
strides and scalar IR before execution. CPU runs use buffer-based Linalg and
flat ABI descriptors, lower through loops to LLVM and execute the JIT. Metal
uses verified same-index or normalized broadcast device IR and exact owned
per-input/output metadata. A bounds guard protects identity/zero/row/column
loads; each input retains its own buffer size. Empty output validates first,
returns its exact shape with no values, clears dumps and requires no device/JIT.

Old literal and flat APIs retain their behavior and equal-length rules. Inputs
are borrowed, read-only and may overlap; outputs own independent data and shape
after caller inputs and compiler/device resources are destroyed. There is no
CPU fallback for nonempty Metal execution.

See [CPU usage](shaped-tensor-cpu.md) and [Metal usage](shaped-tensor-metal.md).

## What M3-E adds

Successful shaped numeric test runs now require both an explicit `shape` and a
`result` or `compare` assertion. A sole expected `error` remains valid. Values
alone, shape alone, success-only assertions and missing shape checks cannot
produce a passing shaped test. Configuration validation includes unselected
cases and unavailable-device runs before any pipeline starts. Flat and raw IR
pipeline contracts are retained.

The acceptance matrix has 30 cases / 64 runs in three grouped source files.
Programs, complete argument configuration and independent expected shape/values
remain together in each `.ml` case; no shared fixture state is introduced.

| File under `test/acceptance_m3/` | Cases | Runs | Purpose |
| --- | ---: | ---: | --- |
| `semantics.ml` | 9 | 22 | Arity 1/2/3, `[N]`/`[1,N]`/`[N,1]`, `[2,3]`/`[3,5]`, row/column/outer/singleton, unused/constant/reordered/shared scalar inputs; two IR runs and two complete MSL checks |
| `grid-boundaries.ml` | 15 | 30 | Precisely 1/W-1/W/W+1/2W+3 flattened outputs for widths 4/32/256, distributed across arity and shape forms |
| `empty.ml` | 6 | 12 | Rank-1/rank-2 empty results, either/both zero axes, nonempty inputs, unused empty arguments and constants |

Boundary counts are exactly `[1,3,4,5,11]`, `[1,31,32,33,67]` and
`[1,255,256,257,515]`. Each Metal boundary run uses the corresponding explicit
width. `2W+3` proves multiple groups plus a padded tail. Related cases are grouped
by behavior; this is not the Cartesian product of every semantic dimension.

Input streams use a fixed 32-bit LCG with seed `0x5c0e0003`, independent streams
per argument, displaced MIN/MAX/-1/0/1 anchors and full-width signed values.
`generate_m3.py` imports the independent set-based shape and nested row/column
oracle from `generate_broadcast.py`. It imports no production shape/indexing
helper, parser, compiler output or backend. Scalar instruction lists are
specified separately from source expressions; masking follows each operation.
Hand anchors cover indexing and overflow, including polynomial results `[-15,1]`
for MAX/MIN. Read-only `--check` verifies anchors and exact generated contents
and file set without writing.

`test/reference/msl/broadcast-row-column.msl` and `broadcast-singleton.msl` are
complete hand-authored shader expectations embedded as `text-equal` assertions.
They check all buffer/count slots, uint types, global index/guard, positive
row/column divisors, singleton zero, every loaded/scalar SSA dependency and the
output index. They are not captured or refreshed from actual exporter output.
GPU IR checks and extraction/export preservation run without device access.

Runner mutations verify missing value/shape assertions on CPU and Metal,
success-only runs, invalid metadata hidden by filters/device skips, unknown
fields, missing pipelines, wrong expected error stages, shape-only mismatches,
and valid compare-plus-shape runs. Strict no-device tests require all six empty
cases to succeed and a nonempty Metal case to fail.

## Complete requirement/evidence map

| Gate | Evidence |
| --- | --- |
| Multiple independent source cases, lexical parsing, filtering and isolation | [M3-T](m3-t-acceptance.md), `grouped_spec_contracts`, `runner_contracts` |
| Rank-1/rank-2 shape-bearing production CPU execution and ownership | [M3-A](m3-a-acceptance.md), `shaped_cpu_contracts`, `shaped_input/same_shape.ml` |
| Exact trailing-axis/zero/singleton inference and CPU indexing | [M3-B](m3-b-acceptance.md), independent nested-loop fixtures, unused/constant and empty contracts |
| Same-shape GPU extraction, owned count metadata and real/stub launch contract | [M3-C](m3-c-acceptance.md), `shaped_gpu_contracts`, `shaped_gpu_stub_contracts`, shared runtime checks |
| Row/column/outer GPU execution with different buffer sizes | [M3-D](m3-d-acceptance.md), `broadcast_gpu/`, `BroadcastGPUContracts.cpp` |
| Exact shape and numeric reference / CPU / real GPU agreement | All positive shaped fixtures plus M3-E's seeded matrix; mandatory shape/value schema |
| MIN/MAX/-1/0/1, wrapping and fixed-seed full-width data | Independent oracle anchors, M3-E semantic/grid cases and retained M1/M2/M3 matrices |
| 1/W-1/W/W+1 and multiple groups plus tail | M3-E exact grid matrix; prior literal/external/broadcast boundaries retained |
| Same/overlapping input storage and independent output data/shape | CPU/direct real GPU contracts, unequal-buffer broadcast overlap, escaping results and canaries |
| Rank/extents/storage/name/incompatible-axis errors | `shaped_input/invalid.ml`, broadcast negative fixtures, pre-device stage assertions |
| Host index/byte/capacity/stride and GPU u32/padded-grid/device-buffer bounds | Metadata-only synthetic CPU/GPU/runtime contracts without forged spans or huge allocations |
| Malformed views/maps/captures/effects/output reads/arithmetic promises | `ShapedCPUContracts.cpp`, production wrapper/region validation |
| Malformed device shape/count metadata and index IR | Real/stub `ShapedGPUContracts.cpp` and `BroadcastGPUContracts.cpp`, retained exporter negative IR |
| Intended index computation visible in full shader and IR | Two M3-E complete MSL goldens plus previous exact GPU IR/MSL checks |
| Flat API equal-length compatibility and previous regressions | Existing M1/M2 fixtures, direct flat/CPU/GPU contracts, independent fixture checks |
| CPU/IR, forced no-device, actual stub and strict real Metal | Full CTest, explicit stub binaries, no-Metal suite and strict Apple M1 runs |

Earlier milestone evidence remains historical; current results below validate
all retained regressions at this revision. No prior fixture/run was removed.

## Environment and commands

Apple M1 (8-core GPU), arm64, macOS 15.5, Homebrew LLVM/MLIR and clang 22.1.8.
The existing CMake build and Python 3 are used. Real Metal commands require
device access outside the desktop sandbox.

```sh
cmake --build build --target tester tensor_interfaces metal_runtime_checks metal_stub_checks cpu_input_contracts gpu_input_contracts gpu_input_stub_contracts grouped_spec_checks shaped_cpu_contracts shaped_gpu_contracts shaped_gpu_stub_contracts -j3
python3 test/reference/generate_m3.py --check
./build/tester --suite test/acceptance_m3 --require-metal
./build/tester --suite test/acceptance_m3 --no-metal
ctest --test-dir build --output-on-failure
./build/tester --require-metal
./build/tester --no-metal
```

| Check | Result |
| --- | --- |
| New M3-E subset, actual Metal required | 64 passed, 0 skipped, 0 failed; 3 files / 30 cases |
| New M3-E subset, Metal disabled | 40 passed, 24 expected device skips, 0 failed |
| Two complete MSL expectations and GPU IR preservation checks | Passed without a device; paired numeric cases passed on Apple M1 |
| CTest, including direct ownership/negative IR/runtime/explicit stub/runner contracts and six independent fixture checks | 18/18 passed |
| Complete strict Metal suite | 842 passed, 0 skipped, 0 failed; 114 files / 412 cases |
| Complete Metal-disabled suite | 558 passed, 284 expected device skips, 0 failed |

The baseline had 111 files / 382 cases / 778 runs. M3-E adds exactly three
source files, 30 cases and 64 runs. No previous source fixture or run is removed.
The Metal-disabled skips are nonempty GPU execution/compilation runs explicitly
forced unavailable; CPU, IR/export, empty, stub and expected-error coverage
remain active. The strict run has no device skips.

M3-T and all M3-A/B/C/D/E checks are satisfied at this revision. M3 is accepted
locally; feature PR merge status is a separate review step.

## Measured and enforced limits

A direct Metal API query on this Apple M1, compiling the complete
`broadcast-row-column.msl` expectation, reported:

| Limit | Value / enforcement |
| --- | --- |
| Device `maxBufferLength` | 8,589,934,592 bytes (8 GiB), checked independently for every input/output buffer |
| Device `maxThreadsPerThreadgroup.width` | 1,024 |
| Representative pipeline `maxTotalThreadsPerThreadgroup` | 1,024; runtime queries the actual compiled pipeline on every call |
| Default launch width | min(256, device/pipeline limit), therefore 256 for this pipeline |
| Metal input/output element counts | At most UINT32_MAX (4,294,967,295), before allocation/dispatch |
| Total padded threads | At most 2^32, making the final global index at most UINT32_MAX |
| CPU/host count and canonical stride/byte products | Checked against signed host index, size_t byte range and vector capacity |

Reported buffer limits are validation ceilings, not an allocation guarantee;
no multi-gigabyte allocation is attempted by acceptance tests. Device-only
limits apply when nonempty execution actually needs Metal. Fixed host/target
count and launch-option validation still runs for empty results.

## Limits and completion boundary

Rank is limited to 1 or 2, with dense contiguous row-major i32 storage and the
existing binary `+ - *` grammar. Negative/dynamic extents, rank 0 or >2, arbitrary
strides/views, transpose/slicing, scalar external parameters, mixed dtypes,
floats, matmul and reductions are excluded. Parentheses/unary minus/general let
expressions are not added to the external-input grammar. Arithmetic remains
exact wrapping i32; host/GPU counts and launch limits reject unsupported ranges.

Calls remain synchronous with per-call translation and compilation. No caching,
persistent sessions, resource pooling or performance improvement is claimed.
Plain launch metadata must remain paired with its verified source/entrypoint.
M4 float numerical policy/execution and M5 reductions remain separate work;
M3 acceptance does not claim those capabilities. Local roadmap/planning files
remain excluded from the PR.
