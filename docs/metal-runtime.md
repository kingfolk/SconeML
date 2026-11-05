# One-shot Metal execution

`runTensorLiteralGPU(source, gpuSource, options)` now runs the complete production path:

```text
source -> shared frontend -> letalg.tensor_map -> GPU kernel IR
       -> Metal exporter -> Metal compilation -> uniform dispatch -> host vector
```

The existing source API and optional generated-source output remain available. `MetalLaunchOptions` is an optional third argument. Each call creates its own context, exported source, Metal library, pipeline, buffers and commands, waits for completion, and returns an owning `std::vector<int32_t>`. There is no cache or persistent device state. The former recursive expression emitter is removed.

## Compiler/runtime boundary

`GPURunner.cpp` retains the limited host-literal recovery step: the program must contain exactly one map whose output is the top-level result. Input values come from the translated literal's constant stores. The compiler-facing `lowerTensorMapToGPU` and `exportMetalKernel` functions remain independent of literal recovery and device APIs.

`runMetalKernel` accepts exported source, the returned entrypoint, input/output spans, an element count, and launch options. Input/output capacities must cover the count. Runtime allocation and readback cover only that prefix. Buffers 0 and 1 are separate shared Metal allocations; buffer 2 receives the unsigned 32-bit element count. The function looks up the supplied entrypoint rather than hardcoding its name.

The source frontend still rejects empty tensor literals. At the lower runtime boundary, count zero is a no-op: no device lookup, compilation, zero-length buffer allocation, or dispatch occurs. This behavior is available on the non-Apple stub as well.

## Launch geometry and validation

The default width is `min(256, pipeline limit, device x-dimension limit)`. A nonzero `threadsPerThreadgroup` overrides it and must fit both limits. Test configuration exposes this as positive `threads-per-group`; zero is reserved for automatic selection in the C++ API.

For count `N` and width `W`, the runtime dispatches `N / W + (N % W != 0)` threadgroups via `dispatchThreadgroups`. Every group has width `W`, including the last. Exported MSL computes:

```cpp
uint index = group_id.x * group_size.x + local_id.x;
if (index < count) {
  // Load, compute and store one element.
}
```

For `N = 5, W = 4`, group 0 handles indices 0–3, and group 1 has indices 4–7. Index 4 executes; indices 5–7 skip. A nonuniform final group would invalidate this index formula, so `dispatchThreads` is not used.

`planMetalLaunch` is a pure C++ helper shared by the real runtime and stub. It checks count fits u32, host capacities, byte multiplication, buffer-size limits, group width and the final padded u32 index. A padded total of 2^32 threads is allowed because its last index is UINT32_MAX. Larger totals are rejected. Device and pipeline limits are checked before buffer allocation and command submission. Metal resource/command creation failures, shader compilation failures, missing entrypoints and unsuccessful command completion throw exceptions; host output is copied only after successful completion.

## Validation

```sh
cmake --build build --target tester tensor_interfaces metal_runtime_checks metal_stub_checks cpu_input_contracts -j3
ctest --test-dir build --output-on-failure
./build/tester --no-metal
./build/tester --suite test/tensor --require-metal
./build/metal_runtime_checks
```

Data files retain explicit expected results and CPU comparisons for polynomial, negative and overflow cases. Identity and width-4 dispatch cases cover lengths 1, 3, 4, 5 and 11. The actual executed MSL is checked for exporter-specific builtins and SSA temporaries. An invalid-width data case verifies launch errors reach the caller. No-device runs explicitly skip GPU execution; `--require-metal` makes those skips fail.

`metal_runtime_contracts` uses synthetic sizes to cover capacity, device limits and u32 launch boundaries without allocating huge buffers. It also invokes the real runtime with zero count and checks compilation/entrypoint-error propagation when a Metal device is available. These are C++ API contracts rather than a second table of compiler programs.

The broader fixed-seed arithmetic/reference matrix remains subsequent milestone work.

F4 local validation on Apple M1 with LLVM/MLIR 22.1.8: the complete `--require-metal` suite passed 101 runs with zero skips/failures; all four CTest entries passed. Forced `--no-metal` passed 90 runs and explicitly skipped 11 device runs. The stub was also compiled separately with the same launch/empty-runtime contract checks.
