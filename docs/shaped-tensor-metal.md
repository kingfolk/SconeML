# Rank-1/rank-2 tensor execution on Metal

`runShapedTensorGPU` in `src/tensor/GPURunner.h` synchronously evaluates one
expression over 1–3 named dense row-major i32 inputs with trailing-axis
broadcasting. It returns owned values and the inferred rank-1/rank-2 shape.

```cpp
#include "src/tensor/GPURunner.h"

std::vector<int32_t> a{1, 2, 3, 4, 5, 6};
std::vector<int32_t> b{10, 20, 30};
std::vector<int32_t> c{100, 200};
std::vector<sconeml::tensor::ShapedTensorArgument> arguments{
    {"a", a, {2, 3}}, {"b", b, {3}}, {"c", c, {2, 1}}};
std::string msl;
auto result = sconeml::tensor::runShapedTensorGPU("a + b + c", arguments, &msl, {4});
// result.shape == {2, 3}
// result.values == {111, 122, 133, 214, 225, 236}
```

The optional final `MetalLaunchOptions` selects a positive explicit threadgroup
width, here 4. Default width is capped at 256 and the device/pipeline limit.
The launcher retains uniform one-dimensional padded groups with guarded loads
and stores. It never expands broadcast inputs into output-sized host buffers.

## Shape, ownership and errors

The API shares [CPU shape inference](shaped-tensor-cpu.md): align axes from the
right, treat missing axes as 1, and accept equal extents or singleton expansion.
Zero with one gives zero; zero with two is incompatible. All declared inputs,
including unused arguments and constant-only expression inputs, participate.
For example `[2,1] + [1,3]` returns `[2,3]`, `[2,3] + [3]` reuses a row vector,
and `[2,3] + [2]` fails instead of assuming column orientation.

Name/rank/extent/product/storage validation, canonical strides, the existing
scalar grammar and wrapping i32 arithmetic retain the shared frontend contract.
Inputs are borrowed and read-only and can alias or overlap. Output owns separate
data and shape after inputs, compiler context and Metal resources disappear.
Nonempty execution without a Metal device fails explicitly; no CPU fallback is
used. Explicit non-Apple stub builds retain this behavior.

Empty inferred shapes such as `[0]`, `[0,3]` or `[2,0]` preserve the exact shape,
return no values and clear the MSL dump without kernel extraction/export,
allocation or device access. This works even when an input is nonempty, for
example `[2,1]` with `[1,0]`. Syntax, all input metadata, target host bounds and
launch options still validate before the empty return. No zero-axis division
or remainder is emitted or executed.

All input/output counts must fit u32, even if an unused input or an empty output
would otherwise avoid reads. Nonempty canonical index ranges and padded global
thread indices must fit u32. Host count/byte/vector/stride limits and each actual
device buffer limit are checked before the corresponding allocation/dispatch.

## Index extraction and validation

The frontend builds validated buffer-based Linalg with broadcast affine maps.
`validateShapedTensorGPU` checks that complete wrapper/scalar contract and target
bounds. Identical shapes retain M3-C's TensorMap bridge, old metadata form and
identical legacy MSL. Different compatible shapes use a separately marked
broadcast device form, with direct extraction from validated Linalg scalar IR.
The source module remains unchanged.

For a nonempty output `[R,C]`, the flattened global index is `i`. Input load
indices are normalized to four forms:

| Input mapping | Device index | Bound inside `i < R*C` |
| --- | --- | --- |
| Full dense input, or equivalent singleton/missing-axis identity | `i` | `< inputCount` |
| All input elements collapse to a singleton | `0` | `< 1` |
| Column `[R,1]` | `i / C` | `< R` |
| Row `[C]` or `[1,C]` | `i % C` | `< C` |

Equal input/output counts normalize to identity, preserving cases such as `[3]`
with output `[1,3]`. Output always stores at `i`. Index constants/division/
remainder and one load per input occur inside the existing bounds guard;
repeated expression references reuse the same loaded scalar.

Broadcast IR carries `sconeml.broadcast` with `input_shapes`, output `shape`,
`input_counts` and `output_count`, plus a `sconeml.broadcast_kernel` marker.
Export validates the schema, shapes, inferred output, counts, rank, canonical
stride ranges and u32 limits. It then checks each exact index operation,
positive divisor, global-index dependency, input binding and output store.
Mixed/missing metadata markers, wrong maps, arbitrary affine arithmetic,
index captures and unguarded loads are rejected.

Unsigned division/remainder is allowed only for the normalized index path of
this verified broadcast form. It does not enable division in scalar source
expressions or relax legacy export. Scalar data still allows only local i32
constants and wrapping addition/subtraction/multiplication. Signed guard
comparisons, no-wrap promises and exact-division promises are rejected.

## Runtime metadata

`MetalKernelSource::shapedLayout` owns per-input counts and the expected output
count, independent of MLIR context lifetime. Keep it paired with its validated
source/entrypoint. For the three-input example, counts are `[6,3,2]` and output
count is 6; input byte allocations/copies are respectively 24, 12 and 8, and
output is 24 bytes. The existing M3-C runtime already supports these separate
sizes and exact input-length/output-count validation.

Inputs bind at slots 0..K-1, output at K, and count at K+1. Larger output capacity
is allowed but only the specialized prefix is written. Readback occurs after
successful command completion. Old flat APIs retain their equal-length and
requested-prefix contracts.

## Tests

```sh
cmake --build build --target tester shaped_gpu_contracts shaped_gpu_stub_contracts -j3
python3 test/reference/generate_gpu_broadcast.py --check
./build/tester --suite test/broadcast_gpu --require-metal
./build/shaped_gpu_contracts test/shaped_gpu/semantics.ml test/broadcast_gpu/semantics.ml
./build/shaped_gpu_stub_contracts test/shaped_gpu/semantics.ml test/broadcast_gpu/semantics.ml
ctest --test-dir build --output-on-failure
```

Grouped source fixtures compare GPU values and shapes with CPU and explicit
independent nested-loop expectations. Device-independent IR/MSL checks prove
index operations, guards, bindings and preservation. Direct contracts mutate
metadata and index IR, check target overflow without huge allocations, and
exercise unequal-buffer overlap/lifetime/output canaries on actual Metal.
See [M3-C evidence](m3-c-acceptance.md) and [M3-D evidence](m3-d-acceptance.md).

Rank >2, arbitrary strides/views, floats, reductions, caches and persistent
execution sessions remain outside this feature. Full M3 acceptance is separate.

The complete milestone gate is documented in [M3 acceptance](m3-acceptance.md),
including the seeded matrix, mandatory shape/value test assertions, complete
MSL expectations and retained malformed-IR/runtime/ownership contracts.
