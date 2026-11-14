# M5 acceptance: global/axis sum and map composition

M5 is **accepted locally** on 2026-11-12. PR merge/review status is separate.
[M5-A](m5-a-acceptance.md), [M5-B](m5-b-acceptance.md) and
[M5-C](m5-c-acceptance.md) establish the CPU tree, Metal execution and materialized
broadcast composition. This layer closes independent acceptance and supplies a
runnable `sum_walkthrough` example.

## Observable contract

The public `runSumI32CPU/GPU` and `runSumF32CPU/GPU` APIs execute
`sum(expression)` and `sum(expression, axis=N)` over one to three dense
rank-1/rank-2 host arrays. For `a[2,3]=[1..6]`, `b[3]=[10,20,30]`,
`sum(a * b, axis=1)` returns owned values `[140,320]`, shape `[2]`.
Axis=0 yields `[50,140,270]`, shape `[3]`; global sum yields scalar `460`,
shape `[]`, with one owned value. Rank-0 inputs are not introduced.

Only one nonnegative axis is supported; it is removed from the output shape.
Negative/list/duplicate/out-of-rank/fractional axes, keepdims, nested reduction,
invalid expressions/names/storage/ranks/broadcasts and overflowing literals
fail explicitly. Every input, including unused inputs, is checked before empty
or device handling. `[2,0]` along axis=1 returns two zero identities;
`[0,3]` along axis=1 returns zero output elements. Empty work needs no JIT/device.

I32 sums wrap modulo 2^32. F32 accumulation uses
`f32-sum-tree256-ftz`: fixed consecutive 256-item padded chunks, eight levels of
adjacent-pair RNE binary32 addition, then the same tree over partials until one
result remains. Input subnormals and tiny/zero results normalize to +0. Safe math,
no reassociation/contraction and width 256 are explicit requirements. NaN
payload/sign and signaling status are unspecified; other bits and classifications
follow the exact specified tree. Overflow/invalid arithmetic are handled by the
tree, not skipped. M4's elementwise signed-FTZ policy remains unchanged.

A fixed tree permits exact independent binary32 expectations, rather than relying
on CPU/GPU equality. For finite paths without intermediate overflow, the oracle
also checks the documented size/magnitude/FTZ-aware bound against an exact rational
sum: `gamma_d*S + q*eta*(1+u)^d`, `u=2^-24`, `eta=2^-126`,
`gamma_d=d*u/(1-d*u)`. `d` is eight times the number of passes; `q` counts the
255 additions per padded chunk across passes. Map rounding and original-input
subnormal normalization have their separately documented contracts. No blanket
epsilon, host-double expected sum or arbitrary device-dependent order is used.

## Implementation and lifecycle evidence

CPU emits explicit zero fill and pairwise Linalg reduction iterators, lowers to
loops/LLVM and executes the generated C ABI through JIT. A host C++ sum loop is
not substituted. The public IR builder rejects nonempty/raw-IR modules.

Metal generates a canonical shader and owned checked dispatch metadata from the
validated sum plan. Initial axis-0 loads are strided and axis-1 loads contiguous;
later partial matrices are contiguous. Guarded inactive lanes write identity,
all lanes reach each group barrier, and one partial is stored per group. For
65,537 inputs the actual device executes `65537 -> 257 -> 2 -> 1`, requiring
three separate dispatches. Different groups are never synchronized by a
threadgroup barrier. Ordinary serial compute encoders and tracked buffers
establish dependencies; precise-lifetime retention lasts through command
completion. Dtype/bytes/profile/source/order/count/capacity/size/width/limits are
checked, and pipeline/command errors propagate without CPU fallback.

Composition materializes through the corresponding M4 CPU or GPU API and keeps
its owned intermediate alive through sum. The first GPU implementation has a
map readback and sum upload; it claims no fusion or persistent-device residency.
Direct named-input sums avoid an unnecessary map. Input/result aliasing and
lifetime contracts are checked with actual overlapping host spans.

## Independent reference and fixture map

`test/reference/generate_sum.py` does not import production parser/indexing,
execute the compiler, or read backend results. It independently selects logical
coordinates/axes, evaluates separate map instruction lists, wraps unsigned
integers and uses M4's independent exact integer/Fraction binary32 arithmetic.
Its sum tree is independently implemented in Python. Always-enabled anchors
remain active under Python `-O`, including wrapping, empty identities, FTZ,
nonassociative cancellation and the rational error bound. Seeded finite streams
use `0x5c0e0005` with LCG multiplier 1664525 and increment 1013904223.

| Fixture group | Cases / runs | Evidence |
| --- | --- | --- |
| `sum_cpu/` | 25 / 25 | CPU global/axis/rank-0, explicit identity/reduction IR, wrapping/float/empty/error cases |
| `sum_gpu/` | 35 / 61 | Real CPU/Metal expected arrays, counts 1/255/256/257/513, multi-group/dispatch and barrier/source/error checks |
| `map_sum/` | 16 / 32 | K=1/2/3, unused input shape, broadcasts, requested example, empty outputs/identities for both dtypes |
| `acceptance_m5/numerics.ml` | 35 / 70 | Exact 2/3/255/256/257/511/512/513/1025/65537 i32 counts, 257-row/column strides, finite f32 through 4097 items, rational bounds, NaN/overflow/FTZ/zero/cancellation across groups, composed K=3 axis/global sums |
| `acceptance_m5/invalid.ml` | 10 / 20 | Invalid storage/rank/names/arity/broadcast/axis/syntax, nested sum and overflowing f32 literals even on empty inputs |
| `acceptance_m5/shaders.ml` | 2 / 2 | Complete independently hand-authored i32/f32 shaders, every binding/guard/scratch/tree/barrier |
| **M5 total** | **123 / 210** | All numeric runs have independent values and shape; composition also compares CPU/Metal |

The 10 generated files are self-contained source/config/assertion fixtures;
ordinary execution does not run Python. `--check` validates anchors, bound cases,
exact contents and exact file sets without writes. The full shaders in
`test/reference/msl/sum-i32.msl` and `sum-f32.msl` are hand-authored expectations,
never refreshed from compiler output.

## Negative and direct API checks

Direct CPU/real-GPU/explicit-stub targets cover C ABI, owned data/shape after
input/JIT destruction, changed inputs with unchanged IR/MSL, actual overlapping
map inputs, metadata/source/byte/capacity errors, missing/reordered/corrupt
partial passes, synthetic padded/u32/stride bounds and unsupported raw IR.

Runner contracts deliberately corrupt numeric values, required rank-0/axis
shapes, zero sign, f32 profile and full shader source and require failure.
Nonempty `--no-metal --require-metal` fails; empty dimensions and zero outputs
pass, invalid source fails at frontend, and invalid width fails at GPU lowering
before any device skip. All 20 final negative source runs pass without a device.
These checks preserve M1-M4's independent shape/value/error/device contracts.

## Environment and final verification

Host/device: **arm64 Apple M1**, **macOS 15.5**, LLVM/MLIR/Homebrew Clang
**22.1.8**, Release. Metal uses safe math, precise FP32 functions and source
contraction disabled. Actual pipeline compilation, all partial dispatches and
numeric readback were executed on this device.

| Check | Result |
| --- | --- |
| All CMake targets | Built successfully; legacy deprecated-builder warnings remain |
| `ctest --test-dir build --output-on-failure` | **27/27 passed** |
| `./build/tester --require-metal` | **1250/1250 passed, zero skips**, 135 files / 654 cases |
| Final `test/acceptance_m5 --require-metal` | **92/92 passed, zero skips** |
| Final acceptance with `--no-metal` | **59 passed, 33 explicit device skips, zero failures** |
| Independent generator `--check`, with Python and Python `-O` | All 10 files/anchors/bound checks passed |
| Runner mutation/device-policy checks | Passed |
| `sum_walkthrough` / `sum_walkthrough --metal` | Both produced `[140,320]` with expected shapes/map/dispatch plan |
| `sum_walkthrough --ir` / `--msl` | Ran successfully and exposed actual map/sum IR and sum source |

The 1250 full runs comprise 1040 pre-M5 runs plus 25/61/32/92 runs from M5-A/B/C/D.
Real-device local evidence is separate from GitHub-hosted CI device availability.

## Remaining boundaries

Evidence is for the named device/toolchain; other targets require conformance
verification. Numerical order is fixed at width 256; arbitrary workgroup widths,
reassociation, arbitrary rounding, NaN payload preservation and throughput claims
are excluded. Map fusion, GPU-resident composition, dynamic shapes, views,
rank>2 inputs, keepdims/axis lists, mixed precision, max/min/product, scans,
softmax, matmul, autodiff, caches and persistent sessions remain separate work.
