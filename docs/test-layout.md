# Source test layout

Source tests are grouped by the behavior they prove. Use a descriptive file for
the topic and `(* @case NAME *)` blocks for individual inputs, boundary points or
rejections. Each case retains its own program, complete configuration and
assertions. Select it with `tester --file PATH --case NAME`; add `--run NAME`
when needed. See [testing](testing.md) for the format and device policies.

## Thematic source files

Paths below are relative to `test/`.

| File | Purpose | Cases |
| --- | --- | ---: |
| [tensor/arithmetic-semantics.ml](../test/tensor/arithmetic-semantics.ml) | Literal identity, signed results and i32 wrapping | 3 |
| [tensor/dispatch/threadgroup-boundaries.ml](../test/tensor/dispatch/threadgroup-boundaries.ml) | Literal dispatch below, at and above a threadgroup boundary | 4 |
| [tensor/dispatch/launch-validation.ml](../test/tensor/dispatch/launch-validation.ml) | Reject unsupported launch widths | 1 |
| [external_input/expression-semantics.ml](../test/external_input/expression-semantics.ml) | Subtraction, constants, precedence, whitespace and unused arguments | 5 |
| [external_input/errors/expression-syntax.ml](../test/external_input/errors/expression-syntax.ml) | Reject unsupported or malformed expressions | 10 |
| [external_input/errors/argument-validation.ml](../test/external_input/errors/argument-validation.ml) | Reject invalid names, missing arguments and unbound names | 5 |
| [external_input/errors/empty-input-validation.ml](../test/external_input/errors/empty-input-validation.ml) | Validate expressions even when input is empty | 2 |
| [external_input/metal/threadgroup-boundaries.ml](../test/external_input/metal/threadgroup-boundaries.ml) | External-input Metal dispatch and tail guards across widths and lengths | 10 |
| [external_input/metal/launch-validation.ml](../test/external_input/metal/launch-validation.ml) | Validate external-input launch widths before device selection | 1 |
| [multi_input/arithmetic-and-binding.ml](../test/multi_input/arithmetic-and-binding.ml) | Arithmetic, declaration order, repeated and unused operands | 6 |
| [multi_input/two-input-length-boundaries.ml](../test/multi_input/two-input-length-boundaries.ml) | Two-input execution across lengths | 6 |
| [multi_input/three-input-length-boundaries.ml](../test/multi_input/three-input-length-boundaries.ml) | Three-input execution across lengths | 6 |
| [multi_input/empty-inputs.ml](../test/multi_input/empty-inputs.ml) | Empty two/three-input execution without a device | 2 |
| [multi_input/errors/argument-validation.ml](../test/multi_input/errors/argument-validation.ml) | Arity, uniqueness, names and unbound variables | 6 |
| [multi_input/errors/length-validation.ml](../test/multi_input/errors/length-validation.ml) | Mismatched lengths, including empty and unused inputs | 3 |
| [acceptance/source/literal-semantics.ml](../test/acceptance/source/literal-semantics.ml) | Independent literal arithmetic expectations and empty-literal rejection | 5 |
| [acceptance_m2/matrix/single-input-launch-boundaries.ml](../test/acceptance_m2/matrix/single-input-launch-boundaries.ml) | Independent one-input oracle and launch boundary matrix | 22 |
| [acceptance_m2/matrix/two-input-launch-boundaries.ml](../test/acceptance_m2/matrix/two-input-launch-boundaries.ml) | Independent two-input oracle and launch boundary matrix | 22 |
| [acceptance_m2/matrix/three-input-launch-boundaries.ml](../test/acceptance_m2/matrix/three-input-launch-boundaries.ml) | Independent three-input oracle and launch boundary matrix | 22 |

Examples:

```sh
./build/tester --file test/external_input/errors/expression-syntax.ml
./build/tester --file test/multi_input/arithmetic-and-binding.ml --case binding-order --run metal --require-metal
./build/tester --file test/multi_input/two-input-length-boundaries.ml --case length-257 --no-metal
./build/tester --file test/acceptance_m2/matrix/three-input-launch-boundaries.ml --case w32-n033 --run metal --require-metal
```

Root-level legacy files intentionally remain independent: they verify the old
single-program annotation format. `tensor/polynomial.ml`,
`external_input/polynomial.ml`, `external_input/identity.ml` and
`multi_input/alias.ml` remain focused standalone fixtures used by direct API or
runner schema contracts. M2's standalone lifetime, overlap, constant and unused
fixtures also retain their existing reader contracts. Raw `.mlir` files retain
their single-module format; this migration does not add grouped IR parsing.

When adding a related source regression, prefer a named case in the matching
thematic file. Create a new file when its purpose differs, or when a direct API
contract or legacy compatibility check needs a standalone fixture. Generated
acceptance files must be edited through their Python generator.

## Consolidation validation (November 7, 2026)

Against the initial M3-T local layout, 75 source files were consolidated into
16 thematic files and the three M2 matrix files were renamed. Source files fell
from 99 to 40; the complete discovered suite fell from 158 to 99 physical files.
The complete suite still contains 221 logical cases and 536 runs.

A migration comparison accounted for every prior source file: all 75 migrated
program/configuration/assertion payloads match their previous text, apart from
case markers and separating blank lines; the three renamed matrices match
exactly; the other 21 source files match exactly. No runs, input arrays, expected
values, structural checks or negative cases were removed. Runner contract
selections now specify file and case. Temporary mutation tests extract their
specific case from the grouped fixture before corrupting it.

Validation used the existing Apple M1 / macOS 15.5 / LLVM 22.1.8 build:

| Check | Result |
| --- | --- |
| CTest, including runner/API/explicit stub contracts and both oracle checks | 11/11 passed |
| Complete suite, actual Metal required | 99 files, 221 cases; 536 passed, 0 skipped, 0 failed |
| M1 independent fixture check | 13 files, unchanged 17 logical cases / 46 runs |
| M2 independent fixture check | 7 files, unchanged 70 logical cases / 207 runs |
| Migration payload and file-set comparison | All 99 previous source files accounted for |

Real Metal execution uses tool permission escalation because the sandbox hides
the device. This is a file-organization change; no test-speed improvement is
claimed.
