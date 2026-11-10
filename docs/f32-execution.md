# Shaped f32 execution

M4-A implements `runShapedTensorF32CPU` in `src/tensor/CPURunner.h`.
`ShapedTensorF32Argument` borrows `span<const float>` values and copies a dense
row-major rank-1/rank-2 shape; `ShapedTensorF32Result` owns values and shape.
One to three uniquely named inputs use the existing trailing-axis broadcasting
rules, including empty outputs and overlapping read-only host spans.

```cpp
std::vector<float> a{1, 2, 3, 4, 5, 6}, b{0.25f, -0.5f, 1.5f};
std::array<sconeml::tensor::ShapedTensorF32Argument, 2> args{{
    {"a", a, {2, 3}}, {"b", b, {3}}}};
auto result = sconeml::tensor::runShapedTensorF32CPU("a * 0.5 + b", args);
// shape [2,3], values [0.75, 0.5, 3, 2.25, 2, 4.5]
```

The separate entry point avoids ambiguous existing empty `{}` calls. All
inputs/results are f32; there are no implicit casts, mixed dtypes, division,
transcendentals, f16, reductions, views, caches or sessions. Expressions support
identifiers, finite signed decimal/scientific literals and `+`, `-`, `*`.
Multiplication binds first; equal-precedence operators associate left to right.
Parentheses, unary variable operators and nonfinite source literals are rejected.
Input arrays can contain infinities and NaNs. Decimal constants convert directly
to binary32 with LLVM APFloat, avoiding an intermediate host double; constants
rounding to infinity are rejected even for empty output.

## Numerical contract: f32-ftz

The profile is part of the verified function IR, not a fast-math option.

1. Convert each input and constant subnormal to zero with the same sign.
2. Evaluate each source arithmetic operation in binary32, round to nearest,
   ties to even, without reassociation or multiply-add contraction.
3. Flush tiny arithmetic results to zero with the same sign, detecting
   tininess **before rounding** (exact magnitude below `2^-126`). In particular,
   `minNormal * nextDown(1)` yields zero, even though rounding alone would yield
   `minNormal`. Integer exponent/significand checks enforce this on CPU and
   Metal. Identity programs still normalize input subnormals.
4. Preserve signed zeros and infinity signs; overflow produces signed infinity.
   Invalid arithmetic and NaN inputs produce NaN; NaN payload/sign and signaling
   exceptions are not specified. No exception trapping/status contract is made.

Supported finite results, including zeros and infinities, are compared exactly
by their binary32 bits; NaNs compare by classification. No epsilon is used.
The CPU runner requires the host round-to-nearest environment and rejects other
rounding modes. Integer bit tests and selects implement signed FTZ explicitly;
the validator checks every normalization sequence, local operand, operation and
fast-math flag before lowering. Float data never uses integer host storage.

Empty output validates syntax, names, shapes and literals before returning an
owned empty vector/shape, without JIT and with an empty LLVM dump. Host values
never become translation-time constants. Existing i32 entry points retain their
wrapping arithmetic and grammar.

## Unified fixtures

`shaped-f32-cpu` executes the production API; `shaped-f32-ir` exposes verified
Linalg IR. Each run requires `profile: f32-ftz` and explicit `dtype: f32` for each
argument. Arguments retain `name`, `shape` and `values`. Float values/results can
be decimal, `inf`, `-inf`, `nan`, or quoted `0xXXXXXXXX` raw binary32 bits.
Raw bits preserve zero signs and allow subnormal/NaN payload probes. `result`
assertions use the exact/classification policy above, `compare` requires an
earlier f32 run, and numeric execution requires a shape assertion. Integer
fixture parsing/assertions remain exact i32. Unknown profiles/dtypes and cross
dtype comparisons are configuration failures, not expected compiler errors.
