# Sum execution

M5-A adds `runSumI32CPU` / `runSumF32CPU` in `src/tensor/Sum.h`.
Inputs use the existing borrowed shaped tensor argument types; `SumResult<T>`
owns values and shape, including explicit rank-0 outputs.

```cpp
std::vector<int32_t> a{1,2,3,4,5,6};
std::array<sconeml::tensor::ShapedTensorArgument,1> args{{{"a",a,{2,3}}}};
auto rows = sconeml::tensor::runSumI32CPU("sum(a, axis=1)",args);
// values [6,15], shape [2]
auto all = sconeml::tensor::runSumI32CPU("sum(a)",args);
// values [21], shape {} -- a scalar, not a rank-1 result.
```

## Shape and source contract

`sum(expression)` reduces all elements. `sum(expression, axis=N)` accepts one
nonnegative axis smaller than the inferred rank. Rank-1/rank-2 inputs remain the
only supported inputs; an axis is removed from the output shape (`keepdims` is
unsupported). Rank-1 axis=0 and global sums produce rank-0 output, represented
by shape `{}` with exactly one owned value. Axis lists, negative axes, duplicate
axes, other keywords, nested sums and trailing source are rejected.

M5-C accepts one to three uniquely named inputs and the existing elementwise
expression grammar with trailing-axis broadcasting. The map operand is
validated and its shape inferred before sum-axis/empty/device handling.

The identity is zero. `[2,0]` along axis=1 produces `[0,0]` with shape `[2]`;
`[0,3]` along axis=1 produces no elements with shape `[0]`. Global empty sum is
one scalar zero. Source, input shape/storage and axis validation occur even for
empty execution. No JIT is created and the LLVM dump is cleared for these cases.

## Numerical contract

I32 addition wraps modulo 2^32. F32 uses the explicit
`f32-sum-tree256-ftz` profile, with binary32 accumulation and a fixed order:

1. Normalize input subnormals/zeros to positive zero.
2. Partition each logical reduction lane into consecutive chunks of 256,
   padding incomplete chunks with zero.
3. Sum adjacent pairs `(0,1), (2,3), ...`, then pairs of those partials, for
   eight levels. Each add is RNE binary32, without fast math/reassociation;
   tiny results and signed zeros normalize to positive zero.
4. Repeat the same process over chunk partials until one result remains.

CPU and Metal must implement this same tree; a launch-width choice cannot
silently change numerical order. CPU requires the RNE host environment.
Overflow yields signed infinity, invalid additions yield NaN; NaN payload/sign
and signaling status are unspecified. All non-NaN expectations use exact bits,
with independent tree expectations; NaNs use classification. Equality between
CPU/GPU alone is not an acceptance oracle. M4 elementwise signed FTZ remains
unchanged; sum's positive-zero identity is a separate documented contract.

For a finite lane with no overflowing intermediate, let `S` be the exact sum
of absolute normalized inputs, `u=2^-24`, `eta=2^-126`, `d` the tree depth and
`q` the number of tree additions including padding. A conservative absolute
error bound relative to the exact rational sum is

`gamma_d*S + q*eta*(1+u)^d`, where `gamma_d=d*u/(1-d*u)`.

The first term bounds binary32 rounding along a path; the second bounds FTZ
losses and their subsequent amplification. Here each dispatch contributes eight
levels, and `q=255*sum(chunks_per_dispatch)`. This is size/magnitude aware and
includes cancellation. Nonfinite/overflow paths instead follow the exact tree's
classification rules. No blanket epsilon or approximation to a host-double sum
is used for acceptance. Original subnormal input loss is accounted for separately
if comparing against unnormalized inputs.

## IR and tests

The CPU builder emits an explicit zero `linalg.fill` and buffer-based
`linalg.generic` operations with parallel/parallel/reduction iterators for each
two-element pair. Padding, input indexing and owned intermediate buffers are
explicit. Standard Linalg -> loops -> LLVM lowering executes through the JIT;
no host C++ sum loop substitutes for compiled execution. Flat C ABI buffers
avoid introducing rank-0 input support.

The unified runner supports `sum-i32-cpu`, `sum-f32-cpu`, `sum-i32-ir` and
`sum-f32-ir`. Every argument has shape/storage; f32 arguments require `dtype:
f32`, and f32 runs require `profile: f32-sum-tree256-ftz`. Successful execution
requires both expected values and shape; only sum accepts shape `[]`.
`test/reference/generate_sum.py --check` verifies independent fixtures read-only,
including exact integer wrapping and rational binary32 addition/tree anchors.

The IR follows the [MLIR Linalg iterator and identity model](https://mlir.llvm.org/docs/Dialects/Linalg/).

## Metal reduction (M5-B)

`runSumI32GPU` / `runSumF32GPU` in `src/tensor/SumGPU.h` execute the same
source and fixed tree. Explicit width is either default/zero or 256; other
widths are rejected even for empty execution. The checked `SumMetalProgram`
owns plan, dtype, byte size, profile, source and dispatch dependencies. Validation
reconstructs the canonical program and rejects altered source/metadata, buffer
length/type errors and byte limits before device work.

Each threadgroup loads up to 256 items into shared scratch, initializes inactive
lanes with zero, and adds adjacent pairs at strides 1,2,4,...128. Every lane
participates in each barrier. One partial is written per group. Subsequent
passes use a contiguous partial matrix and repeat until one result per output
remains. Axis=0 initially uses strided column loads; axis=1 uses contiguous rows.

The runtime uses separate ordinary serial compute encoders in one command
buffer and tracked shared buffers, preserving producer/consumer dependencies.
A precise-lifetime owner retains all partial buffers until completion; no
threadgroup barrier is treated as a cross-group barrier. Host/device byte limits,
256-lane and static scratch limits, pipeline/command errors are checked.
Explicit GPU execution has no CPU fallback. Empty-output and empty-reduction
identities succeed without compiling/accessing a device, with an empty dump.

`sum-i32-metal` / `sum-f32-metal` expose production execution;
`sum-i32-msl` / `sum-f32-msl` expose canonical source without a device.
Source/schema/axis/launch validation precedes device skips in the test adapter.
The [Metal compute encoder model](https://developer.apple.com/library/archive/documentation/Miscellaneous/Conceptual/MetalProgrammingGuide/Compute-Ctx/Compute-Ctx.html)
provides the serial encoding and threadgroup execution basis; ordinary tracked
resources are used, not untracked heaps or concurrent encoders.


## Materialized map-to-sum composition (M5-C)

```cpp
std::vector<int32_t> a{1,2,3,4,5,6}, b{10,20,30};
std::array<sconeml::tensor::ShapedTensorArgument,2> args{{
    {"a",a,{2,3}}, {"b",b,{3}}}};
auto result=sconeml::tensor::runSumI32CPU("sum(a * b, axis=1)",args);
// map [10,40,90,40,100,180], then result [140,320], shape [2].
```

The GPU API accepts the same arguments/source. The existing M4 frontend checks
all argument names, storage and broadcast shapes, including unused inputs;
axis validation uses the complete inferred map shape. Nontrivial maps are
materialized through the matching production shaped CPU or GPU API, and an
owned vector remains alive until sum finishes. A single direct named-input sum
avoids an unnecessary map invocation. Empty maps still validate syntax/literals
and every declared argument before returning identities without target work.

This is synchronous materialization, not kernel fusion. On Metal, the M4 map
completes and returns owned host values before the sum uploads/executes its
partial-reduction dispatches. There is no hidden CPU evaluation/reduction in an
explicit GPU request; a missing device fails nonempty map/sum execution. No
performance or persistent-buffer claim is made. The optional LLVM/MSL dump and
sum IR adapters describe the sum stage; inspect the operand separately with
M4's shaped frontend/IR adapters to examine map lowering.

The float map preserves the M4 signed-FTZ scalar contract. Its results then enter
the explicitly distinct fixed-tree sum profile, including positive-zero identity.
The reduction error bound applies to those mapped/normalized input values;
map rounding is accounted for by its own per-operation contract/oracle.

## Complete evidence and runnable learning example

[M5 acceptance](m5-acceptance.md) records the independent tree/reference model,
size/magnitude bounds, real three-dispatch execution, full shader expectations,
negative cases and final environment/results. Build/run `sum_walkthrough` for
the requested `sum(a * b, axis=1)` example; use `--metal`, `--ir`, `--msl` to
inspect the actual backend, map/reduction IR and sum shader.
