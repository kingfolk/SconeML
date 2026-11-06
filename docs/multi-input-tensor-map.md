# Two and three input tensor maps

M2-C extends the existing `runTensorCPU` / `runTensorGPU` APIs to 1–3 named,
read-only, contiguous i32 tensors. Every declared input must have the same
length. One output vector owns independent storage. The source expression
grammar, wrapping i32 arithmetic, per-call JIT and literal APIs are unchanged.

```cpp
#include "src/tensor/CPURunner.h"
#include "src/tensor/GPURunner.h"
#include <vector>
using namespace sconeml::tensor;

std::vector<int32_t> a{1, 2, 3};
std::vector<int32_t> b{10, 20, 30};
std::vector<int32_t> c{4, 5, 6};
std::vector<TensorArgument> arguments{{"a", a}, {"b", b}, {"c", c}};
auto cpu = runTensorCPU("a * b + c", arguments);
auto gpu = runTensorGPU("a * b + c", arguments);
// Both are {14, 45, 96}. All three inputs remain unchanged.
```

For element i, this computes `a[i] * b[i] + c[i]`. It does not concatenate arrays,
broadcast lengths, or create intermediate tensor kernels. Names bind to scalar
values loaded at the current index. Declaration order determines parameter
order, not alphabetical order or order of first appearance in the expression.

## Validation and ownership

Zero arguments, more than three arguments, duplicate/invalid names and unknown
expression variables are errors. Length mismatch is an error even when an input
is unused or one array is empty. All-empty equal-length calls still validate
and then return empty without CPU JIT or Metal device execution.

Read-only inputs may reference identical or overlapping host storage. Each input
has its own name and map argument; no no-alias promise is attached to inputs.
The source runners always allocate a separate output. The input spans must
remain alive for the synchronous call and are never modified or freed. GPU
execution copies input bytes into independent device buffers; aliasing of
read-only host inputs does not change elementwise semantics.

## Compiler changes

`TensorMapOp` now has variadic `inputs` followed by one output. The verifier
requires 1–3 matching rank-1 i32 inputs/output and one i32 scalar body argument
per input. Output cannot be an input SSA operand. Repeated read-only input SSA
operands are allowed. The region remains isolated and supports only local i32
constants and wrapping add/sub/mul, with a single matching yield.

Memory effects describe one read per input operand and one output write.
Tests inspect the actual OpOperand identity so repeated SSA values do not
accidentally duplicate or hide effects. No operand-segment attribute is needed:
there is one variadic group and a fixed final output operand. Existing
single-input generic IR remains valid without syntax changes.

The frontend sees argument names and lengths only. It creates a function with
K input memrefs and one output memref, and binds each expression variable to its
corresponding scalar map argument. Repeated references reuse that scalar. CPU
lowering loads each input at the loop index and uses IRMapping for all body
arguments. GPU lowering does the same inside its existing unsigned bounds guard.
Cloning preserves shared SSA definitions.

The CPU runner builds K input descriptors and an output descriptor. It invokes
the generated C interface using an exact typed signature for K=1/2/3 rather than
calling a fixed-arity function with a mismatched pointer type. All arrays are
length-specialized per call, as before; no dynamic-shape inference or cache is
introduced.

## Metal signature and runtime

`MetalKernelSource` now includes validated `inputCount` alongside `source` and
`entrypoint`. For K inputs, the binding convention is:

| Slot | Meaning |
| --- | --- |
| 0..K-1 | Input buffers in declaration order |
| K | Independent output buffer |
| K+1 | One unsigned i32 element count |

The exporter validates K input loads at the same index, in parameter order,
followed by supported scalar operations, a store to output and the terminator.
It emits one local per SSA result and returns the arity used by that signature.
K=1 retains the original `input` name, full shader text and 0/1/2 bindings.
K>1 uses `input0`, `input1`, and optionally `input2`.

The generalized runtime receives the exported count and the supplied input span
list. `validateMetalInputs` rejects arity mismatch and checks every input's
capacity before device access, including empty calls. Existing launch validation
checks output capacity, bytes, device/pipeline width and padded u32 indices.
The old single-input runtime overload delegates to the same implementation.
The non-Apple stub shares validation and empty behavior; nonempty requests fail
explicitly without CPU fallback.

Uniform padded threadgroups, the linear index formula, bounds guard, synchronous
completion and readback are unchanged. CPU and GPU source output ownership is
independent of JIT/device resources and input lifetimes after return.

## Tests and local validation

`test/multi_input` contains self-contained source and IR files. Numeric cases
have explicit independent expected arrays and CPU/GPU comparisons for two-input
addition, three-input multiply/add, parameter reordering, repeated/unused inputs,
full-width wrapping arithmetic and empty arrays. Widths 1/4/32/256 and lengths
1/3/4/5/33/257 cover exact and padded groups. Expectations use integer arithmetic
reduced to signed 32-bit patterns, not backend output.

IR cases check complete three-input MSL, load/store/SSA order, shared SSA,
per-operand effects including aliased inputs, and rejection of incorrect body
arity, input count/type/shape, output aliases, captures, effects, layouts and
swapped device loads. `CPUInputContracts` and `GPUInputContracts` additionally
read `alias.ml` and bind two names to the same actual host address; result/input
ownership is checked directly. Runtime/stub contracts test metadata mismatch,
short second/third buffers and valid empty two/three-input calls. The formerly
invalid two-argument source case is retained as a positive unused-input case.

```sh
cmake -S . -B build
cmake --build build --target tester tensor_interfaces metal_runtime_checks metal_stub_checks cpu_input_contracts gpu_input_contracts gpu_input_stub_contracts -j3
ctest --test-dir build --output-on-failure
./build/tester --suite test/multi_input --require-metal
./build/tester --require-metal
./build/tester --no-metal
./build/gpu_input_contracts test/external_input/polynomial.ml test/multi_input/alias.ml \
  test/acceptance_m2/lifetime.ml test/acceptance_m2/overlap.ml
```

On Apple M1 with LLVM/MLIR 22.1.8: the complete strict Metal suite passed 329
runs with zero skips/failures; no-device mode passed 244 runs and explicitly
skipped 85 device runs. CTest passed 9/9. Direct real-device input alias and
ownership contracts passed. The non-Apple stub is explicitly tested on macOS,
which does not constitute a separate Linux platform validation.

M2-D adds the broader generated/reference matrix and direct three-input
lifetime/overlap contracts; see the [M2 acceptance record](m2-acceptance.md) for
the final matrix, commands and observed results. The counts above describe the
M2-C validation baseline. This feature does not add general functions/imports, broadcasting,
more than three inputs, floating point, reductions, caches or async execution.
