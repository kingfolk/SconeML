# Same-shape rank-1/rank-2 tensor execution on Metal

`runShapedTensorGPU` in `src/tensor/GPURunner.h` synchronously evaluates one
expression over 1–3 named i32 inputs with identical dense row-major shapes.
Rank 1 and rank 2 are supported. Results own their values and shape.

```cpp
#include "src/tensor/GPURunner.h"

std::vector<int32_t> a{1, 2, 3, 4, 5, 6};
std::vector<int32_t> b{10, 20, 30, 40, 50, 60};
std::vector<sconeml::tensor::ShapedTensorArgument> arguments{
    {"a", a, {2, 3}}, {"b", b, {2, 3}}};
std::string msl;
auto result = sconeml::tensor::runShapedTensorGPU("a + b", arguments, &msl);
// result.shape == {2, 3}
// result.values == {11, 22, 33, 44, 55, 66}
```

An optional final `MetalLaunchOptions` selects a positive explicit threadgroup
width, for example `runShapedTensorGPU("a + b", arguments, &msl, {4})`.
Default width is capped at 256 and the device/pipeline limit. The launcher uses
uniform one-dimensional padded threadgroups with guarded loads and stores.

## Shape, ownership and error contracts

The API uses the shared [shaped frontend](shaped-tensor-cpu.md), including its
name, rank, extent, checked product/storage, canonical stride, scalar grammar,
wrapping arithmetic and ownership rules. Input spans are borrowed and read-only;
inputs can alias or overlap. The output has separate storage and survives input,
compiler context, JIT and Metal resource destruction. No CPU fallback is used.

M3-C requires all input shapes to be identical. Compatible but different shapes
such as `[2,3]` with `[3]`, `[2,1]` with `[1,3]`, or `[3]` with `[1,3]` are
explicit errors before device availability is considered. CPU broadcasting
continues to work. Identical singleton shapes such as `[1,3]` and `[3,1]` are
supported: constant-zero singleton maps are equivalent to same-index loads
because that output dimension also has extent one.

Empty shapes `[0]`, `[0,3]` and `[2,0]` preserve their exact result shape, validate
syntax, arguments, supported GPU shapes and fixed host launch bounds, and
return no values without kernel extraction, MSL export, allocation or device
access. The optional MSL dump is cleared and empty. Broadcasting remains an
error even for empty output. Nonempty calls without a Metal device throw;
non-Apple builds use the explicit stub and also throw for nonempty execution.

Nonempty count and padded thread indices must fit u32. Input/output byte sizes
and vector capacity are checked on the host; each device allocation is also
checked independently against the actual device buffer limit. Invalid launch
options fail even when output is empty or the device is unavailable.

## Validated extraction and export

`validateShapedTensorGPU` first validates the complete shaped Linalg/wrapper
contract, then requires identical input shapes and checks the target count.
`lowerShapedTensorToGPU` builds a temporary flat TensorMap from the already
validated scalar region and passes it through the existing GPU extraction.
The source module remains unchanged. This is a focused identity-index bridge;
it does not relax the old exporter to accept general Linalg or affine indexing.

A shaped device module carries `sconeml.shaped` with `shape`, `input_counts` and
`output_count`, and its kernel has a `sconeml.shaped_kernel` marker. The exporter
verifies both markers, rank/extents/product/strides, metadata types and arity,
exact input/output counts and the unchanged same-index kernel envelope before
emitting MSL. Missing or inconsistent shaped metadata is rejected. The resulting
MSL is identical to legacy flat MSL for the same scalar expression and arity.

`MetalKernelSource::shapedLayout` owns a `MetalKernelLayout`: a vector of
per-input element counts and the expected output count. It contains no MLIR
references and can outlive the compiler context. The plain layout must remain
paired with the source/entrypoint from its validated export.

`runShapedMetalKernel` requires exact input lengths/arity and an exact requested
output count matching this metadata. Output capacity may be larger; only the
specialized prefix is written. Pure `planShapedMetalLaunch` checks synthetic
lengths/capacities and returns each input's byte count plus output launch data.
The real runtime copies each buffer using its own verified byte count, then
reads output back only after successful command completion. Real/stub host
preflight and empty behavior are shared. Old flat entry points remain wrappers
with their existing prefix/capacity semantics.

## Tests

```sh
cmake --build build --target tester shaped_gpu_contracts shaped_gpu_stub_contracts \
  metal_runtime_checks metal_stub_checks -j3
python3 test/reference/generate_shaped_gpu.py --check
./build/tester --suite test/shaped_gpu --require-metal
./build/shaped_gpu_contracts test/shaped_gpu/semantics.ml
./build/shaped_gpu_stub_contracts test/shaped_gpu/semantics.ml
ctest --test-dir build --output-on-failure
```

The grouped `shaped-input-metal` fixtures compare owned values and shape against
CPU and explicit expectations from the independent nested-loop Python oracle.
`shaped-input-gpu-ir` and `shaped-input-msl` inspect device IR, count metadata,
parameter order, guards and MSL without device access. Direct contracts verify
ownership/aliasing, metadata lifetime and corruption, target bounds, exact
runtime bindings and canaries. See [M3-C acceptance](m3-c-acceptance.md).

GPU broadcasting, arbitrary strides/views, floats, reductions, caches and
persistent execution sessions are outside this feature.
