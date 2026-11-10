# M4 acceptance: typed f32 CPU and Metal execution

M4 is **accepted locally** on 2026-11-10. This is implementation/validation
status; PR merge status is separate. M4-A adds the typed frontend/CPU path,
M4-B adds verified Metal code generation/storage, and M4-C closes independent
numerical acceptance. See [usage and numerical contract](f32-execution.md),
[A evidence](m4-a-acceptance.md) and [B evidence](m4-b-acceptance.md).

## Observable capability and numerical decision

`runShapedTensorF32CPU` and `runShapedTensorF32GPU` execute homogeneous f32
rank-1/rank-2 elementwise expressions with one to three bound host arrays,
fractional/scientific constants, broadcasting and owned float/shape results.
For `a[2,3] * 0.5 + b[3]`, both return the independently expected `[2,3]` result.
The existing flat/shaped i32 APIs retain wrapping arithmetic, ownership and
error contracts.

Native CPU/Metal probes preceded implementation. Safe/precise Metal with
contraction disabled still flushed subnormal operands/results and detected
multiplication tininess before rounding, while the CPU retained subnormals and
could round an exact tiny product up to the minimum normal. The selected
`f32-ftz` profile explicitly resolves this difference with integer bit tests and
exact 48-bit significand checks on both paths. Every input and constant
subnormal becomes same-sign zero; arithmetic with exact magnitude below
`2^-126` becomes same-sign zero before rounding. Other finite arithmetic uses
RNE binary32 at each source operation, without contraction/reassociation.
Constants first round directly to binary32, then normalize. Overflow produces
signed infinity; invalid operations/NaNs produce NaN. NaN payload/sign and
signaling status are unspecified. Zero/infinity signs remain significant.

Every supported non-NaN result is checked by **exact bits**, and NaNs by
classification. No epsilon or tolerance can hide cancellation, signed zeros,
overflow, underflow or an unsupported profile. CPU execution rejects non-RNE
host rounding. No unresolved numerical exception is skipped for acceptance.

## Independent expectations

`test/reference/generate_m4.py` never executes the compiler or imports production
parser/indexing code. It decodes binary32 patterns into exact Python integers
and `Fraction` rationals, evaluates a separate instruction list, detects exact
tininess, and rounds by quotient/remainder with ties-to-even. It uses no host
float/double arithmetic as an oracle. Source expressions and oracle instructions
are specified separately. Broadcast expectations use logical-coordinate
cartesian products and singleton selection, independently of production affine
maps and GPU flattened division/remainder paths.

Always-enabled hand-computed anchors check RNE even/odd ties, separated
multiply-add, exact-minimum-normal controls, positive/negative pre-rounding
tininess, input/intermediate FTZ, cancellation, overflow signs, signed zero,
Inf/NaN combinations and direct decimal conversion. `python3 -O` retains all
checks. Generated files are self-contained source/config/assertion fixtures;
ordinary `tester` never invokes Python or refreshes expectations.

| File under `test/acceptance_m4/` | Cases / runs | Coverage |
| --- | --- | --- |
| `arithmetic.ml` | 22 / 44 | RNE, full 15x15 special-value cross products for +/−/*, signaling/quiet NaNs, signed zeros/Inf, cancellation, underflow/overflow, decimal ties, constants and repeated products |
| `broadcast.ml` | 13 / 26 | Mixed ranks/order, row/column/outer/singleton and same-shape maps, nonsquare matrices and unused input shape participation |
| `grid.ml` | 15 / 30 | Widths 4/32/256, exact counts 1/W−1/W/W+1/2W+3, one/multiple groups and tails |
| `empty.ml` | 6 / 12 | Zero rows/columns and empty broadcasts with nonempty inputs, strict no-device success |
| `magnitude.ml` | 5 / 10 | 257-element full-width seeded input streams, wide exponents, normals/subnormals, exceptional arithmetic and composed expressions |
| `structure.ml` | 1 / 3 | CPU/Metal arithmetic plus a complete independently hand-authored MSL expectation, bindings, guard, constant bits, FTZ sequences and source preservation |
| **Total** | **62 / 125** | Every numeric CPU and Metal run has an independent expected array and shape; Metal additionally compares with CPU |

Seed: `0x5c0e0004`, 32-bit LCG multiplier `1664525`, increment `1013904223`.
`test/reference/msl/f32-add.msl` is a hand-authored full shader embedded in the
structural fixture. It is never regenerated from actual exporter output.
`--check` verifies anchors and exact generated file contents/file set without
writing. Regeneration is explicit and must accompany review of intended changes.

## Contracts and negative evidence

The M4-A/B direct executables cover typed C ABI, disjoint/overlapping input
storage, input immutability, value-independent IR/MSL, metadata/result lifetime
after source/MLIR/JIT destruction, owned shapes/results, empty execution,
nonempty unavailable-device errors, exact capacities/counts and launch limits.
The explicit Metal stub builds on macOS as well. Validation rejects changed or
missing profile/type/byte metadata, unnormalized stores, altered FTZ masks,
fast-math promises, bad literal/syntax/shape/name/arity and dtype mismatches.
All prior integer/shape/ownership/exporter/runtime suites remain required.

Runner contracts deliberately corrupt a finite expectation, zero sign, profile,
dtype, result/shape completeness, comparison target and full shader golden;
each must fail for the expected reason. Filtering out a device run cannot hide
its invalid schema. `--no-metal --require-metal` fails nonempty GPU execution,
but all 12 empty acceptance runs and 12 malformed-input GPU runs pass without
skips. This prevents a missing device or loose comparison from appearing as
numerical success.

## Environment and final results

- Device: **Apple M1**; host: arm64, **macOS 15.5 (24F74)**, SDK **15.5**.
- LLVM/MLIR and Homebrew Clang: **22.1.8**; CMake build: **Release**.
- CPU: verified no-fast-math MLIR/LLVM arithmetic, optimizing JIT at O2, explicit
  integer normalization; caller rounding checked as RNE.
- Metal: `MTLMathModeSafe`, `MTLMathFloatingPointFunctionsPrecise`, exported
  `#pragma STDC FP_CONTRACT OFF`; real library/pipeline compilation and dispatch.

| Command | Result |
| --- | --- |
| `cmake --build build --parallel 3` | All targets built; existing legacy deprecated-builder warnings remain |
| `ctest --test-dir build --output-on-failure` with real device access | **22/22 passed** |
| `./build/tester --require-metal` | **1040/1040 passed, zero skips**, 125 files / 531 cases |
| `./build/tester --suite test/acceptance_m4 --require-metal` | **125/125 passed, zero skips** |
| `./build/tester --suite test/acceptance_m4 --no-metal` | **69 passed, 56 explicit nonempty-device skips, zero failures** |
| `python3 test/reference/generate_m4.py --check` | Anchors and all 6 files / 62 cases / 125 runs verified |
| `python3 -O test/reference/generate_m4.py --check` | Same checks passed with Python assertions disabled |
| Runner configuration/assertion/device-policy contracts | All passed, including deliberate f32 corruptions |

The full suite comprises 842 pre-M4 runs, 28 M4-A runs, 45 M4-B runs and
125 M4-C acceptance runs. These are real-device local results, not a claim that
GitHub-hosted CI has a Metal device. Native scalar probes are reproducible from
`test/reference/probe_f32.mm`; commands and observed bits are in the A evidence.

## Boundaries

Verified device/toolchain evidence is for Apple M1/macOS 15.5/LLVM 22.1.8.
Other targets require their own conformance evidence; no bitwise NaN payload,
exception trapping, arbitrary rounding, gradual-underflow or performance claim
is made. Canonical integer FTZ/product checks add work and code size; this
milestone establishes semantics rather than a throughput target.

Mixed dtypes/promotion, casts, f16, division/transcendentals, rank above two,
dynamic shapes, strides/views, reductions, compilation caches and sessions stay
outside M4. M5 must separately define accumulation order and reduction error.
