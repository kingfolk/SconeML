# M5-A: reduction semantics and CPU evidence

Scope: direct rank-1/rank-2 i32/f32 sum on CPU, global or one explicit axis.
See [usage, fixed-tree contract and size/magnitude-aware error bound](sum-execution.md).
Map composition and real GPU execution belong to the next stack layers.

The 25 self-contained source/structural runs under `test/sum_cpu/` cover global,
rows/columns, rank-0 results, i32 wraparound, fixed-tree f32 cancellation, FTZ,
NaNs/infinities, overflow, both empty-output and empty-reduction-dimension cases,
invalid axes/keywords/source and explicit Linalg identity/reduction iterators.
The independent Python oracle reuses only M4's rational binary32 arithmetic,
never production code or backend output; the sum tree and logical axes are
modeled separately. Always-enabled hand-computed anchors include a cancellation
case that distinguishes the specified tree from a linear fold.

`sum_cpu_contracts` exercises the direct C ABI, data/shape ownership after
input/JIT destruction, changed host values with identical LLVM IR, rank-0 empty
identity/dump behavior, corrupt plan counts/strides, synthetic range errors and
sum-only scalar shape assertions. Existing M1-M4 tests remain required.

Environment: LLVM/MLIR 22.1.8, Release, arm64 macOS 15.5, Apple M1.
Final validation results are appended below.

- All CMake targets built successfully.
- CTest: **24/24 passed**.
- Full strict Apple M1 suite: **1065/1065 passed, zero skips**.
- Sum CPU source/structural suite: **25/25 passed**.
- Direct CPU contracts, oracle `--check`, and Python `-O --check`: passed.
