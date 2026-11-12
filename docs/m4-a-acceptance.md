# M4-A acceptance: typed frontend and CPU execution

This record covers the CPU feature; full M4 additionally requires Metal and
independent end-to-end numerical acceptance. See [usage/profile](f32-execution.md).

## Numerical decision gate

Native scalar arithmetic was probed before defining public f32 behavior using
`test/reference/probe_f32.mm`, compiled on this machine with LLVM/Clang 22.1.8,
`-std=c++20 -O2 -ffp-contract=off -fobjc-arc`, Foundation and Metal frameworks.
Metal used `MTLMathModeSafe`, precise FP32 functions and
`#pragma STDC FP_CONTRACT OFF`. Device: Apple M1.

Observed hexadecimal binary32 results:

| Probe | CPU | Metal |
| --- | --- | --- |
| `(1+2^-23)*(1-2^-23)-1`, separated multiply/add | `00000000` | `00000000` |
| `1+2^-24`, even halfway tie | `3f800000` | `3f800000` |
| `(1+2^-23)+2^-24`, odd halfway tie | `3f800002` | `3f800002` |
| minimum normal times 0.5 | `00400000` | `00000000` |
| minimum subnormal times 2 | `00000002` | `00000000` |
| `-0 * 1` | `80000000` | `80000000` |
| infinity times zero | NaN | NaN |
| maximum finite times 2 | `7f800000` | `7f800000` |
| adjacent minimum normals subtracted | `00000001` | `00000000` |
| minimum normal times nextDown(1) | `00800000` | `00000000` |

CPU and Metal differ for subnormals even without fast math. M4 therefore selects
explicit signed FTZ for both backends rather than claiming gradual underflow or
hiding the difference with tolerances. NaN payloads also differed, so only NaN
classification is promised. The compiled CPU expression fixtures independently
exercise these same properties, including a subnormal intermediate subsequently
multiplied back toward the normal range. M4-B must prove the compiled Metal path.

Sources: [Metal Shading Language specification, compiler options and numerical
compliance](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf),
[Metal safe math](https://developer.apple.com/documentation/metal/mtlmathmode/safe),
[LLVM floating point semantics](https://llvm.org/docs/LangRef.html#floating-point-environment).
Metal permits denormal flushing and default contraction; selecting safe math
alone does not establish cross-backend binary32 equality.

Reproduce the native probe on macOS with a visible Metal device:

```sh
clang++ -std=c++20 -O2 -ffp-contract=off -fobjc-arc test/reference/probe_f32.mm \
  -framework Foundation -framework Metal -o /tmp/sconeml-f32-probe
/tmp/sconeml-f32-probe
```

## Verification

The grouped `test/f32_cpu/` fixtures contain 28 runs with hand-computed binary32
anchors for fractions, broadcasts, constants, decimal conversion, RNE ties,
noncontraction, signed zero, FTZ inputs/intermediates/results, overflow, NaNs,
infinities, cancellation, empty shapes and malformed source/shape/name cases.
Direct `f32_cpu_contracts` checks the C ABI, ownership after input/JIT destruction,
input alias/overlap, value-independent IR, empty validation, unsupported rounding,
corrupt profile/FTZ/fast-math IR, and fixture schema/bitwise assertion behavior.

Validation commands/results are recorded after the final feature checks below.

- `cmake --build build --parallel 3`: all targets built (Release, LLVM/MLIR
  22.1.8; existing legacy deprecated-builder warnings remain).
- `ctest --test-dir build --output-on-failure`: **19/19 passed** with Metal.
- `./build/tester --require-metal`: **870/870 passed, zero skips**, including
  **28/28** new f32 CPU source runs and all **842** prior runs.
- `./build/f32_cpu_contracts test/f32_cpu/semantics.ml`: passed.
- Expanded native probes confirmed pre-rounding tininess for both signs and
  the exact-minimum-normal control. Explicit exponent/significand normalization
  closes this CPU/Metal difference without tolerance.
