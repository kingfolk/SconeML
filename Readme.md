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

## Tensor IR and CPU runner

SconeML now has a local tensor IR for dynamic rank-1 `f32` buffers.
`letalg.tensor_map` contains a scalar region evaluated for every input element
and writes the result to the matching output element. The supported scalar
operations are `arith.constant`, `arith.addf`, `arith.subf`, `arith.mulf`, and
`arith.divf`.

The CPU runner lowers the operation through SCF and memref to the LLVM dialect,
then executes native machine code through MLIR's execution engine:

```text
letalg.tensor_map -> scf.for + memref -> LLVM dialect -> native CPU JIT
```

Run the CPU assertion with:

```sh
cmake -S . -B build \
  -DLLVM_DIR="$(brew --prefix llvm)/lib/cmake/llvm" \
  -DMLIR_DIR="$(brew --prefix llvm)/lib/cmake/mlir" \
  -DCMAKE_CXX_COMPILER="$(brew --prefix llvm)/bin/clang++"
cmake --build build --target tensor_targets -j4
./build/tensor_targets
```

The test evaluates `x*x + 2*x + 1` and asserts the native result against the
expected tensor. Tensor literals in the source parser, multidimensional shapes,
reductions, fusion, allocation, and device placement are follow-ups.

On macOS, the GPU runner translates the same scalar region to Metal Shading
Language, compiles it for the default Apple GPU, dispatches the kernel, and
compares every result element against both the expected tensor and the CPU JIT:

```text
letalg.tensor_map -> Metal Shading Language -> native Apple GPU
```

Run the cross-target test on a Metal-capable machine with `./build/tensor_targets`.
Other platforms keep the CPU test active and use a GPU runtime stub until a
CUDA, ROCm, or SPIR-V backend is added.

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
