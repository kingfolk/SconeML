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

Maps must satisfy the existing verifier: matching rank-1 i32 memrefs, distinct input/output SSA operands, a single isolated scalar block, and a matching yield. Scalar computation supports i32 constants and wrapping addition, subtraction, and multiplication. Implicit captures, memory effects, and no-wrap promises are rejected.

GPU extraction additionally requires identity memref layouts and the default host memory space. Strided/offset layouts and non-default memory spaces produce a diagnostic rather than being silently reinterpreted as contiguous buffers.

The source memref may have a static or dynamic length. This describes the IR API only; it does not extend source-language input syntax. The generated device signature uses an explicit count and does not embed the source length.

## Generated kernel

| Argument | Device IR type | Contract for future execution |
| --- | --- | --- |
| 0: input | `memref<?xi32, #gpu.address_space<global>>` | Contiguous i32 input, read-only by convention |
| 1: output | Same as input | Contiguous i32 output, separate allocation |
| 2: count | i32 | Count interpreted as unsigned bits |

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

Build and run without a GPU:

```sh
cmake -S . -B build
cmake --build build --target tensor_gpu_lowering -j3
ctest --test-dir build -R '^tensor_gpu_lowering$' --output-on-failure
```

To inspect the generated device IR alongside the test output:

```sh
./build/tensor_gpu_lowering \
  test/kernel_lowering/shared_square.mlir \
  test/kernel_lowering/shared_square_gpu.mlir --dump
```

The handwritten [source fixture](../test/kernel_lowering/shared_square.mlir) computes a square once and adds it to itself. The [expected fixture](../test/kernel_lowering/shared_square_gpu.mlir) is a structural assertion: the test compares types, attributes, operations, and SSA dependencies while ignoring source locations. It also explicitly checks SSA sharing and verifies that the source module was not modified.

Additional tests cover print/parse round trips, capture rejection, unsupported scalar operations, invalid yields, incompatible layouts/memory spaces, and extraction from frontend-generated polynomial, subtraction, and identity maps.

`SconeMLTensorGPU` contains the production transformation. The test links the LetAlg implementation and shared frontend alongside it, with no CPU runner, ExecutionEngine, or Metal frameworks. CI builds and runs the new test as part of the tensor checks.

## Execution status

This change stops at verified device IR. It does not export MSL, generate a host launch, or execute the GPU kernel. The existing Metal runner continues to use its original expression-to-MSL implementation. Its replacement requires the subsequent exporter and runner integration work.
