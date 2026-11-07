# SconeML

A MLIR recipe for ocaml. Taste of ML language from MLIR perspective. Hope taste like best baked scone!

**This is a ongoing project. It's a proof of concept for using MLIR for functional language syntax and will not be suitable for production in any time. following features may be done in future**

language syntax:
- [ ] `mut` keyword allow variable mutable
- [ ] variant, tuple, list type

compiler opts:
- [x] Alpha transformation to solve name conflict
- [x] Variable capture as formal parameter
- [ ] Inline let/lambda to some extent
- [ ] Lower dialect to llvm and native

I would like to achieve these above features based on optimization passes or extending LetAlg dialect we already have. I would also like to finish a compact runtime data structure design

- [ ] efficient stack frame

## Tensor IR and runners

SconeML supports rank-1 integer tensor literals such as
`let ts = [1, 2, 3] in ts * ts + 2 * ts + 1`.
`letalg.tensor_map` contains a scalar region evaluated for every input element
and writes the result to the matching output element. The current frontend
generates `i32` constants and element-wise addition, subtraction, and multiplication.

The CPU and GPU runners share `TensorFrontend.cpp`, which verifies the generated
IR before target lowering. A tensor map body is isolated from enclosing values
and accepts only i32 constants and wrapping addition, subtraction, and
multiplication. Captures, memory effects inside the body, and other scalar
operations are rejected. The map itself declares an input read and an output
write, so it is not treated as a pure operation by optimization passes.

The [unified test runner](docs/testing.md) reads programs, pipeline configuration,
and assertions from the same `.ml` or `.mlir` file. Source test files can group
independent programs with `(* @case NAME *)`; select one with `--case NAME`.
The [test layout](docs/test-layout.md) groups related cases by purpose. Run compiler tests without a
GPU with `cmake --build build --target tester tensor_interfaces metal_runtime_checks metal_stub_checks cpu_input_contracts gpu_input_contracts gpu_input_stub_contracts grouped_spec_checks` followed by
`./build/tester --no-metal`.

An independent [GPU kernel extraction](docs/gpu-kernel-lowering.md) API converts
a verified tensor map into `gpu.module` / `gpu.func` with explicit buffer/count
arguments, thread indexing, and guarded loads/stores. This device-IR path is
tested without a GPU. An independent [MSL exporter](docs/metal-exporter.md)
converts that restricted kernel IR into Metal source, with per-SSA temporaries
and explicit unsigned integer semantics. The [Metal runtime](docs/metal-runtime.md) executes these exported shaders
with uniform, padded threadgroups and checked input/output/count bindings.

The CPU runner lowers the operation through SCF and memref to the LLVM dialect,
then executes native machine code through MLIR's execution engine:

```text
letalg.tensor_map -> scf.for + memref -> LLVM dialect -> native CPU JIT
```

Run the CPU assertion with:

```sh
cmake -S . -B build \
  -DLLVM_DIR="$(brew --prefix llvm@22)/lib/cmake/llvm" \
  -DMLIR_DIR="$(brew --prefix llvm@22)/lib/cmake/mlir" \
  -DCMAKE_CXX_COMPILER="$(brew --prefix llvm@22)/bin/clang++"
cmake --build build --target tester tensor_interfaces metal_runtime_checks metal_stub_checks cpu_input_contracts gpu_input_contracts gpu_input_stub_contracts grouped_spec_checks -j3
./build/tester --suite test/tensor --run cpu
```

The tests compare the CPU result against expected values for several tensor
expressions, including negative results and 32-bit overflow.

On macOS, the GPU runner translates the same scalar region to Metal Shading
Language, compiles it for the default Apple GPU, dispatches the kernel, and
compares every result element against both the expected tensor and the CPU JIT:

```text
letalg.tensor_map -> GPU kernel IR -> Metal Shading Language -> native Apple GPU
```

Run cross-target cases with
`./build/tester --suite test/tensor --require-metal` on a Metal-capable machine.
The `tensor-metal` pipeline executes the GPU IR/exporter path; `tensor-msl`
and `metal-compile` also allow checking export and compilation independently.
The GPU runner currently accepts a top-level tensor literal expression with one
tensor map; it extracts the literal values from the translated IR. Other
platforms keep the CPU test active and use a GPU runtime stub until a CUDA,
ROCm, or SPIR-V backend is added. Multidimensional tensors and reductions are
future work.

The [M1 acceptance matrix](docs/m1-acceptance.md) adds an independent 32-bit
reference, reproducible source/IR fixtures and real CPU/Metal result checks.
Python 3 is required for the read-only fixture consistency check in CTest.
The IR execution adapters are test-only; the source API still accepts literals.

The CPU and GPU runners additionally accept [one to three external host tensors](docs/external-tensor-input.md):
`runTensorCPU("xs * xs + 1", arguments)` or `runTensorGPU(...)` accepts named
i32 data and returns a new host vector. The expression is specialized to the supplied length; input values
are not embedded in IR. GPU execution uses the same frontend and existing Metal runtime.
Equal-length inputs can be combined as `a * b + c`; see [multi-input maps](docs/multi-input-tensor-map.md). Existing literal APIs continue to work.

## ML in MLIR dialect

- ML's let sytle
```ocaml
let x = 1 in let y = 2 in x + y
```

in LetAlg dialect
```llvm
module {
  func.func @test_function() {
    %0 = letalg.let (){
      %c1_i32 = arith.constant 1 : i32
      %c2_i32 = arith.constant 2 : i32
      %2 = arith.addi %c1_i32, %c2_i32 : i32
      letalg.yield %2 : i32
    } -> i32 attributes {declCnt = 2 : i32}
    letalg.yield %0 : i32
  }
}
```

This is dumping before passes. It shows the primitive form of letalg. We can still see the scope info like two constants are defined inside `let` op's region.

- lambda
```ocaml
let f x = x + 10 in f 2
```

in LetAlg dialect
```llvm
module {
  func.func @test_function() {
    %0 = letalg.let (){
      %2 = letalg.lambda "f" (%arg0: i32){
        %c10_i32 = arith.constant 10 : i32
        %5 = arith.addi %arg0, %c10_i32 : i32
        letalg.yield %5 : i32
      } -> (i32) -> i32
      %c2_i32 = arith.constant 2 : i32
      %3 = "letalg.apply"(%2, %c2_i32) : ((i32) -> i32, i32) -> i32
      letalg.yield %3 : i32
    } -> i32 attributes {declCnt = 1 : i32}
    letalg.yield %0 : i32
  }
}
```

`lambda` is a callable op and `apply` is a call op.

### Elementwise tensors

The tensor frontend recognizes a bracket literal as a rank-1 integer tensor.
Its elementwise body supports `+`, `-`, `*`, and parentheses:

```ocaml
let ts = [1, 2, 3] in ts * ts + 2 * ts + 1
```

It infers `ts` as `tensor<i32>`, materializes `memref<3xi32>` input/output
buffers, and translates the scalar body to `letalg.tensor_map`. Tensor fixtures
use `@letalg:tensor` to assert this generated LetAlg IR.

- currying

```ocaml
let f x y = x + y + 10 in f 2
```

in LetAlg dialect
```llvm
module {
  func.func @test_function() {
    %0 = letalg.let (){
      %2 = letalg.lambda "f" (%arg0: i32,%arg1: i32){
        %5 = arith.addi %arg0, %arg1 : i32
        %c10_i32 = arith.constant 10 : i32
        %6 = arith.addi %5, %c10_i32 : i32
        letalg.yield %6 : i32
      } -> (i32, i32) -> i32
      %c2_i32 = arith.constant 2 : i32
      %3 = "letalg.apply"(%2, %c2_i32) : ((i32, i32) -> i32, i32) -> ((i32) -> i32)
      letalg.yield %3 : (i32) -> i32
    } -> (i32) -> i32 attributes {declCnt = 1 : i32}
    letalg.yield %0 : (i32) -> i32
  }
}
```

`%0 = letalg.let` return type is `(i32) -> i32`. This `let` op take function type `(i32, i32) -> i32` and only provide the first parameter and return the curried function.

## Passes

There only a few rewriting/optimization passes right now. It's in very primitive stage. An example of rewriting before and after

Current passes mainly works on closure and scope, like erase scope(`let`) and capture as parameters of closure. It will made easy to lower to next step low level dialect.

input is following. lambda `f` has a capture variable from outer closure.
```ocaml
let a = 1 in let f x = x + a + 10 in f 2
```

__before__. Following is initial form of letalg representation, which is nested. This nested representation is good expressive for input in natural because ml's syntax is deeply nested.
```llvm
func.func @test_function() {
  %0 = letalg.let (){
    %c1_i32 = arith.constant 1 : i32
    %2 = letalg.lambda "f" (%arg0: i32){
      %5 = arith.addi %arg0, %c1_i32 : i32
      %c10_i32 = arith.constant 10 : i32
      %6 = arith.addi %5, %c10_i32 : i32
      letalg.yield %6 : i32
    } -> (i32) -> i32
    %c2_i32 = arith.constant 2 : i32
    %3 = "letalg.apply"(%2, %c2_i32) : ((i32) -> i32, i32) -> i32
    letalg.yield %3 : i32
  } -> i32
  letalg.yield %0 : i32
}
```

__after__. All `let` ops are eliminated. The op structure is less nested but in a flat way.

```llvm
func.func @test_function() {
  %c1_i32 = arith.constant 1 : i32
  %0 = letalg.lambda "f" (%arg0: i32,%arg1: i32){
    %3 = arith.addi %arg1, %arg0 : i32
    %c10_i32 = arith.constant 10 : i32
    %4 = arith.addi %3, %c10_i32 : i32
    letalg.yield %4 : i32
  } -> (i32) -> i32
  %c2_i32 = arith.constant 2 : i32
  %1 = "letalg.apply"(%0, %c1_i32, %c2_i32) : ((i32) -> i32, i32, i32) -> i32
  letalg.yield %1 : i32
}
```
