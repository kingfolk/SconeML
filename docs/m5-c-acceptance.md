# M5-C: broadcast map-to-sum composition

`sum(expression[, axis=N])` now accepts the M4 elementwise grammar and one to
three shaped host arguments. Axis inference uses the complete broadcast map
shape. CPU and GPU materialize through their corresponding production APIs,
then execute fixed-tree reduction; the GPU path does not fall back to CPU.
The synchronous host-owned intermediate stays alive through reduction.
See [usage and materialization boundaries](sum-execution.md).

`test/map_sum/semantics.ml` contains 16 cases / 32 CPU/Metal runs with independent
expected map/reduction results: global/axis-0/axis-1, the requested
`sum(a * b, axis=1)` example producing `[140,320]`, one/two/three used inputs,
unused input shape participation, overlap-shaped inputs, empty reduction and
zero outputs, each for i32/f32. The independent reference evaluates a separate
instruction list with wrapping integers or exact rational binary32 operations,
then applies its separate logical-axis/tree oracle.

Direct CPU/real-GPU/stub contracts additionally use actual overlapping borrowed
spans from the same host allocation, check input preservation and output lifetime,
and require nonempty composition to fail on a stub. A/B invalid-name expectations
now exercise the shared M4 frontend's explicit unknown-variable diagnostic.

Environment: Apple M1, macOS 15.5, LLVM/MLIR 22.1.8, Release.

- All CMake targets built; CTest **26/26 passed** with real device access.
- New composition suite **32/32 passed**, zero skips.
- Full strict Apple M1 suite **1158/1158 passed**, zero skips.
- Independent wrapping/rational map/tree checks passed under Python and Python `-O`.
