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

M5-A requires one named input as the operand. M5-C will add expressions with
multiple inputs and broadcasting through the existing elementwise APIs.

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
