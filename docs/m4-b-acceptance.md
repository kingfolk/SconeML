# M4-B acceptance: f32 Metal code generation and storage

This layer depends on M4-A and implements the Metal side of the documented
[f32-ftz contract](f32-execution.md). Full M4 numerical acceptance follows in
M4-C; this record alone does not declare the milestone complete.

F32 always extracts a normalized shaped/broadcast kernel, with integer indexing
and explicit f32 operations/storage. Export verifies the complete scalar
normalization grammar, input index paths, shape/count metadata, homogeneous
f32 dtype, four-byte elements and `f32-ftz` profile. A mismatched type, byte size,
profile, fast-math promise or unnormalized store is rejected. F32 constants use
bit-pattern construction and shaders disable contraction. Actual pipeline
compilation uses safe math and precise FP32 functions on macOS 15+, or disabled
fast math on older SDK-supported systems.

The runtime shares only checked byte allocation/copy mechanisms. Public i32 and
f32 boundaries reject the other dtype; exported layout fields survive MLIR/context
destruction. Existing integer shaders retain their exact text and wrapping
semantics. Empty float results validate syntax/shapes/launch without generating
MSL or requiring a device.

## Verification

`test/f32_gpu/` contains 45 source/structural runs. Hand-computed expectations
cover rank-1/rank-2 row/column broadcasts, three inputs, fractional/scientific
constants, RNE ties, noncontraction, input/intermediate/output FTZ, pre-rounding
tininess, cancellation, signed zeros, overflow, NaNs/infinities, empty output
and malformed inputs. Numeric Metal runs check independent explicit expectations
and CPU comparisons; the structural fixture checks float versus integer/ulong
values, profile metadata and source preservation without Metal.

`f32_gpu_contracts` and the explicitly linked `f32_gpu_stub_contracts` cover
owned metadata/source after context destruction, bad dtype/bytes/profile/counts,
short inputs/output, launch limits, integer/float API mismatches, corrupt IR,
empty and unavailable-device behavior, direct ABI, changed inputs with unchanged
shader, input overlap and result ownership/lifetime.

Device: **Apple M1**. Toolchain: LLVM/MLIR **22.1.8**, Release build.

Validation commands/results are recorded after the final feature checks below.

- `cmake --build build --parallel 3`: all targets built.
- `./build/tester --suite test/f32_gpu --require-metal`: **45/45 passed**, zero skips.
- `./build/tester --suite test/f32_gpu --no-metal`: **30 passed, 15 explicitly
  skipped**, zero failures; empty execution and malformed input cases still ran.
- `ctest --test-dir build --output-on-failure`: **21/21 passed** with real Metal.
- `./build/tester --require-metal`: **915/915 passed**, zero skips, retaining all
  842 pre-M4 and 28 M4-A runs.
- Direct real/stub f32 GPU contracts passed, including actual shader pipeline
  compilation and numeric execution on Apple M1.
