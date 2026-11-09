# Same-shape rank-1/rank-2 CPU execution

`runShapedTensorCPU` in `src/tensor/CPURunner.h` evaluates one expression over
1–3 named, dense row-major i32 inputs. It returns owned values and an owned shape.
Each call translates, lowers and JIT-compiles synchronously.

```cpp
#include "src/tensor/CPURunner.h"

std::vector<int32_t> a{1, 2, 3, 4, 5, 6};
std::vector<int32_t> b{10, 20, 30, 40, 50, 60};
std::vector<sconeml::tensor::ShapedTensorArgument> args{
    {"a", a, {2, 3}}, {"b", b, {2, 3}}};
auto result = sconeml::tensor::runShapedTensorCPU("a + b", args);
// result.shape == {2, 3}
// result.values == {11, 22, 33, 44, 55, 66}, in row-major order
```

## Validation and ownership

Every input requires rank 1 or 2, nonnegative signed 64-bit extents, and exactly
the number of values given by its checked shape product. Names follow the
existing external-input identifier rules and must be unique. All declared
inputs, including unused ones, must have exactly the same shape. Equal storage
counts alone do not imply equal shapes. Rank-0/rank-3 inputs, dynamic extents,
missing/extra values, incompatible shapes and index/byte/vector-capacity
overflow throw before allocation or JIT.

Expressions use the existing external-input grammar: names, nonnegative i32
constants, and binary `+`, `-`, `*`, with the existing precedence. Every scalar
operation uses wrapping i32 arithmetic. Parentheses, unary minus, let bindings,
floats and other operators are outside this API's grammar.

Host spans are borrowed and read-only. They may alias or overlap and must remain
valid until the call returns. The runner never modifies or frees input storage.
The result owns both its data and shape after the inputs, compiler context and
JIT are destroyed. Later calls cannot invalidate an earlier result.

Shapes such as `[0]`, `[0,3]` and `[2,0]` are valid. Empty output still validates
all names, shapes, storage, syntax and generated IR, then returns the exact
shape and zero values without JIT. Its optional LLVM dump is cleared and empty.
Canonical stride products are checked even when a leading zero extent makes
the total count zero; an unrepresentable trailing stride is rejected.

Broadcasting and shaped GPU execution are not implemented in this feature.
There is no shaped GPU entry point or implicit CPU fallback. The existing
`TensorArgument`, literal APIs and flat `runTensorCPU`/`runTensorGPU` retain
their equal-length contracts and do not acquire broadcasting.

## Production IR path

`translateShapedTensorExpression` accepts only names, counts and shapes; host
values never enter the frontend. It builds a `shaped_tensor_expression` function
with K flat rank-1 input memrefs and one flat output memref, marked with
`llvm.emit_c_interface`. Each flat argument is reinterpreted as a static,
zero-offset, canonical rank-1/rank-2 view. These views do not allocate memory.

A buffer-based `linalg.generic` uses one identity indexing map per input/output
and only parallel iterators. The scalar region reuses the existing expression
builder. It accepts input scalars, ignores the uninitialized output scalar and
yields one i32 value. Repeated input references reuse their scalar argument.

`validateShapedTensorExpression` verifies the complete narrow wrapper envelope:
flat ABI/counts, operand ordering, view shapes/strides/offsets/uses, identity
maps, parallel iterators, local i32 scalar constants/add/sub/mul, absence of
captures/effects/output reads, and wrapping arithmetic flags. It runs after
translation and again before CPU lowering. This does not expose arbitrary
Linalg, affine maps, strides or views as an execution API.

The production pipeline converts Linalg to SCF loops, lowers affine operations
and uses the existing SCF/control-flow, memref/arith/func-to-LLVM conversion and
ExecutionEngine. Verification runs after conversion; the JIT rejects any
remaining non-LLVM operations. K=1/2/3 calls use exact descriptor signatures,
without mixed-rank host casts. The old TensorMap lowering remains intact.

## Tests

```sh
cmake --build build --target tester shaped_cpu_contracts -j3
./build/tester --suite test/shaped_input --no-metal
./build/shaped_cpu_contracts test/shaped_input/same_shape.ml
ctest --test-dir build --output-on-failure
```

The grouped fixtures keep programs, input configuration, independently specified
expected values and shape assertions together. They also check frontend Linalg
structure and LLVM elimination. Direct API contracts add ownership, overlap,
specialization and malformed in-memory IR checks. See
[M3-A acceptance](m3-a-acceptance.md) for recorded validation.
