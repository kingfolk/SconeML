# M3-B acceptance: broadcasting on CPU

Date: November 8, 2026 (America/New_York). Baseline: `7869fe5` (M3-A, PR #20).
This records M3-B CPU broadcasting. Shaped GPU execution remains deferred.

## Delivered behavior

`runShapedTensorCPU` now infers trailing-axis broadcasting over every declared
rank-1/rank-2 i32 input and returns owned values and the inferred shape. The
matrix `[2,3]`, row `[3]` and column `[2,1]` example executes `a + b + c` as
`[111,122,133,214,225,236]` with shape `[2,3]`. Outer, singleton, mixed-rank and
zero-extent broadcasts use their original input buffers without expansion.

The frontend returns the checked output shape. Flat ABI descriptors and static
views retain each input's own element count; the output descriptor uses the
inferred product. Buffer-based Linalg uses exact projection/constant-zero input
maps and an identity output map. The production validator re-infers shape from
input views, verifies output equality, per-buffer counts and canonical views,
and rejects any indexing outside the inferred broadcast maps. Existing scalar,
output-read, effect, capture and arithmetic-flag restrictions remain active.

## Independent reference and fixtures

`test/reference/generate_broadcast.py` uses a set-based shape oracle and explicit
nested row/column loops to select input elements. Separately specified scalar
instructions compute wrapping i32 values with Python integers; no production
parser, compiler output, inference helper, runner or backend is imported. Hand
anchors cover row/column addition, outer broadcasting, signed overflow and
zero/singleton rules. Negative anchors reject incompatible axes. `--check`
verifies oracle anchors, the exact generated file set and every file's contents
without writing.

The generated source/configuration/assertions are grouped by purpose:

| File under `test/shaped_input/broadcast/` | Cases | Runs | Coverage |
| --- | ---: | ---: | --- |
| `semantics.ml` | 25 | 26 | Row, column, outer, singleton, all six three-input binding orders, constants, unused inputs, overlap and nonsquare signed matrices |
| `empty.ml` | 8 | 8 | 0/1 rules, either argument empty, both axes zero, missing leading axes and constant-only empty output |
| `invalid.ml` | 8 | 8 | Incompatible axes, zero vs non-singleton, unused/third input validation and syntax/name/storage errors despite empty output |

The additional IR run checks complete three-input indexing maps, distinct ABI
buffer sizes, canonical view metadata, input/output operand order and the exact
scalar block through its yield. The block's output element is never read.

M3-A rejection fixtures retain 21 cases with updated broadcast-era expectations:
shape compatibility now means trailing-axis compatibility. The former deferred
row-broadcast rejection becomes a rank-1 orientation rejection (`[2,3]` vs
`[2]`); valid row broadcasting has explicit independent positive coverage.
The constant-expression rejection now uses genuinely incompatible `[2]` and
`[0]` inputs. Other old flat, literal and same-shape regressions remain active.

Direct C++ contracts additionally verify mixed-rank argument order, result
lifetime, overlapping spans with unequal sizes, all eight empty broadcasts
with cleared LLVM dumps, and inferred outer-product overflow through metadata
without creating huge spans or allocations. Malformed IR checks reject a
MLIR-valid but wrong-row constant index, an invalid singleton-axis index,
output scalar reads and an oversized output shape that generic Linalg alone
permits. The old flat API still rejects unequal input lengths.

## Validation

Apple M1 (8-core GPU), arm64, macOS 15.5, Homebrew LLVM/MLIR and clang 22.1.8.
Real Metal checks run with device access outside the desktop sandbox.

```sh
cmake --build build --target tester tensor_interfaces metal_runtime_checks metal_stub_checks cpu_input_contracts gpu_input_contracts gpu_input_stub_contracts grouped_spec_checks shaped_cpu_contracts -j3
python3 test/reference/generate_broadcast.py --check
./build/tester --suite test/shaped_input --no-metal
./build/shaped_cpu_contracts test/shaped_input/same_shape.ml \
  test/shaped_input/broadcast/semantics.ml test/shaped_input/broadcast/empty.ml
ctest --test-dir build --output-on-failure
./build/tester --require-metal
./build/tester --no-metal
```

| Check | Result |
| --- | --- |
| Independent broadcast oracle anchors and exact generated fixtures | Passed, 3 files |
| Complete shaped source subset | 78 passed, 0 skipped, 0 failed; 5 files / 76 cases |
| Direct shaped CPU ownership, broadcast and malformed-IR contracts | Passed |
| CTest, including explicit stub/API/runner and three independent fixture checks | 13/13 passed |
| Complete suite with actual Metal required | 614 passed, 0 skipped, 0 failed; 104 files / 297 cases |
| Complete suite with Metal disabled | 417 passed, 197 expected device skips, 0 failed |

The baseline had 101 files, 256 cases and 572 runs. This feature adds three
files, 41 cases and 42 runs; retained baseline runs still execute. No speedup
claim is made.

## Limits

The public shaped API remains CPU-only and synchronous, with 1–3 borrowed dense
row-major rank-1/rank-2 i32 inputs. Broadcasting applies only to the shaped API;
old flat CPU/Metal calls still require equal lengths. No shaped GPU entry point,
host buffer expansion, new source grammar, arbitrary maps/views, floats,
reductions, caching or sessions are introduced. This completes M3-B, not the
full M3 milestone. Local planning documents are excluded from the PR.
