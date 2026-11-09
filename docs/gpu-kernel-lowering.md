# Tensor Map to GPU Kernel IR

`lowerTensorMapToGPU` extracts a verified `letalg.tensor_map` into a new device-only MLIR module. This is a compiler transformation, implemented in [LowerTensorToGPU.cpp](../src/conversion/LowerTensorToGPU.cpp). It accepts the map operation directly, without requiring a literal, a particular let nesting, or a source string.

The original host module is not modified. The result contains `gpu.module @kernels` and one `gpu.func @tensor_map` marked as a kernel. Each call returns a separate module, so it does not insert duplicate symbols into the host module.

## Interface and ownership

The declaration is in [LowerTensorToGPU.h](../src/conversion/LowerTensorToGPU.h):

```cpp
mlir::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>>
lowerTensorMapToGPU(letalg::TensorMapOp map);
```

The caller supplies a non-null map in a live MLIRContext. On success, it owns the returned module; the context must outlive both the source and result. On failure, the function emits an MLIR diagnostic and returns failure. A partially generated module is not returned.

The function loads the GPU and supporting dialects into the supplied context as needed. It verifies source IR before inspecting or cloning its body, then verifies the completed device module before returning it.

## Supported source maps

Maps must satisfy the verifier: 1-3 matching rank-1 i32 input memrefs and one output, no output/input SSA alias, a single isolated scalar block with one argument per input, and a matching yield. Read-only inputs may alias. Scalar computation supports i32 constants and wrapping addition, subtraction, and multiplication. Implicit captures, memory effects, and no-wrap promises are rejected.

GPU extraction additionally requires identity memref layouts and the default host memory space. Strided/offset layouts and non-default memory spaces produce a diagnostic rather than being silently reinterpreted as contiguous buffers.

The source memref may have a static or dynamic length. This describes the IR API only; it does not extend source-language input syntax. The generated device signature uses an explicit count and does not embed the source length.

## Generated kernel

| Argument | Device IR type | Contract for future execution |
| --- | --- | --- |
| 0..K-1: inputs | `memref<?xi32, #gpu.address_space<global>>` | Contiguous i32 input, read-only by convention |
| K: output | Same as input | Contiguous i32 output, separate allocation |
| K+1: count | i32 | Count interpreted as unsigned bits |

The generated body has this structure:

```text
index = block_id.x * block_dim.x + thread_id.x
size = index_castui(count)
if index < size:                 // unsigned comparison
  element = load input[index]
  result = cloned_scalar_body(element)
  store result, output[index]
return
```

Thread/block values and indexing use MLIR `index`. Count uses `arith.index_castui` to avoid signed extension when widening a count with its high bit set. Index width remains target-dependent; this transformation does not establish a Metal data layout.

An IRMapping maps the source element argument to the device load result. Each scalar operation is cloned once, and all its consumers use the same mapped SSA result. The map's yield becomes the value stored into the output; `letalg.yield` itself is not copied. The device module contains no host allocations or language operations.

Verification does not prove runtime buffer capacity, allocation non-aliasing, or dispatch arithmetic safety. A future launcher must check capacities against count, require separate allocations, and validate count and padded-grid indices against its u32 indexing contract. A future MSL exporter must define the corresponding pointer, scalar-binding, and index representation.

## Tests

GPU extraction is covered by the [unified data runner](testing.md). Build and run its self-contained case without a GPU:

```sh
cmake --build build --target tester -j3
./build/tester --file test/kernel_lowering/shared_square.mlir
```

The [shared-square file](../test/kernel_lowering/shared_square.mlir) contains the source IR, complete expected GPU IR, SSA capture checks, and expected MSL. The runner compares IR structures, verifies round-trip parsing and input preservation, and checks shared SSA dependencies. Source-language polynomial, subtraction, identity and overflow cases live under test/tensor; invalid maps and layouts have independent data files.

The production transformation remains in SconeMLTensorGPU. A separate small interface executable checks standalone frontend linkage without the CPU runner/ExecutionEngine.

## Execution status

The public GPU runner now uses this extraction API and the independent MSL exporter for actual Metal execution. The `tensor-metal` data pipeline verifies numeric results with uniform padded dispatch; see [Metal runtime integration](metal-runtime.md). Lowering/export-only checks remain runnable without a device.

Multi-input lowering loads every input at the same index inside the guard, maps all scalar block arguments, clones shared SSA definitions once, and stores the result. K=1 retains the original signature. See [multi-input maps](multi-input-tensor-map.md) for the source API, validated ABI and tests.

## Shaped broadcasting

`lowerShapedTensorToGPU` additionally accepts the production shaped Linalg
wrapper. Identical shapes retain the existing TensorMap extraction. Different
compatible rank-1/rank-2 shapes produce a separately marked device module with
shape/count metadata and guarded identity/zero/row/column loads. Scalar IR is
cloned once and each declared input is loaded once per output thread. The host
module is preserved. See [shaped Metal execution](shaped-tensor-metal.md).
