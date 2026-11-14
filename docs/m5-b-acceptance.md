# M5-B: Metal fixed-tree sum evidence

This PR adds global/axis i32/f32 Metal reductions to the M5-A semantics.
See [sum usage and numerical contract](sum-execution.md).

The 61 source/structural runs in `test/sum_gpu/` cover global/row/column loads,
rank-0 output, i32 wrapping, f32 cancellation/FTZ/NaN/Inf/overflow, both kinds of
empty dimension, invalid axes/source, widths at counts 1/255/256/257/513 and
one/two GPU dispatches. Each numeric CPU/Metal run has independent wrapping or
rational-tree expectations. Structural checks inspect scratch, guards, uniform
barriers and the adjacent-pair tree.

`sum_gpu_contracts` and an explicitly compiled `sum_gpu_stub_contracts` verify
canonical dtype/byte/profile/source/dispatch metadata, wrong capacities/device
byte limits, unsupported width, three-stage synthetic dependencies, empty
identities, direct ABI/input immutability and result ownership. The real target
executes on Apple M1 when available; the stub rejects nonempty execution.

The Metal implementation uses safe/precise math, disabled contraction, 256-lane
threadgroups and separate serial encoders over tracked buffers. Precise lifetime
retention keeps all partial resources until the command completes. Errors do not
trigger host reduction fallback. M5-C adds materialized map composition; M5-D
provides broader independent acceptance.

Environment: Apple M1, macOS 15.5, LLVM/MLIR 22.1.8, Release.

- CMake all targets built; CTest **26/26 passed** with real device access.
- New strict GPU suite **61/61 passed**, zero skips.
- Full strict Apple M1 suite **1126/1126 passed**, zero skips.
- Explicit no-Metal suite **41 passed, 20 device skips**, zero failures.
- Direct real/stub GPU contracts and independent oracle check passed.
