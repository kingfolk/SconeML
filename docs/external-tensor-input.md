# One external tensor on CPU

`runTensorCPU` executes an elementwise expression over caller-provided host data.
It accepts exactly one named, contiguous i32 input and returns a new owning host
vector. This is M2-A; external inputs on Metal and multi-input expressions are
not implemented yet. The existing literal CPU/GPU APIs remain available.

```cpp
#include "src/tensor/CPURunner.h"
#include <vector>

std::vector<int32_t> values{-2, 0, 3};
std::vector<sconeml::tensor::TensorArgument> arguments{{"xs", values}};
auto result = sconeml::tensor::runTensorCPU("xs * xs + 1", arguments);
// result == {5, 1, 10}; values is unchanged.
```

## API and ownership

```cpp
struct TensorArgument {
  std::string name;
  std::span<const int32_t> values;
};
std::vector<int32_t> runTensorCPU(
    const std::string &expression,
    std::span<const TensorArgument> arguments,
    std::string *loweredLLVMIR = nullptr);
```

The input data is borrowed until the synchronous call returns. Do not destroy
or concurrently modify its storage during the call. The runner does not free
or write input data. Output is allocated by the runner and remains valid after
both the input storage and JIT engine are destroyed. Modifying output does not
modify input. The named wrapper owns its name, not its tensor storage.

Every nonempty call translates, lowers and creates an LLVM JIT independently.
Calling the same expression with different values or lengths is supported;
there is no compile-once object or cache. The function's memref shape is
specialized to the input length for each call, not represented as dynamic IR.

For zero elements, the frontend still validates the name, expression and map,
then the runner returns an empty vector without creating a JIT. The optional
LLVM dump is cleared and remains empty. A misspelled name or malformed
expression still fails for empty data. Size/index/byte overflow is rejected;
allocation failures propagate normally for otherwise representable large data.

## Expression syntax

The initial grammar is deliberately restricted:

```text
operand (('+' | '-' | '*') operand)*
operand = argument-name | nonnegative-i32-constant
```

Multiplication binds before addition/subtraction; addition and subtraction
associate left to right. ASCII argument names contain letters, digits and
underscore, cannot begin with a digit, and cannot be reserved names such as
`let` or `if`. Spaces, tabs and newlines are accepted. Unknown variables,
missing operands/operators, unsupported characters and out-of-range constants
raise an exception before IR construction or JIT execution.

Negative/extreme host elements are supported independently of constant syntax.
Operators preserve modulo-2^32 i32 semantics. Constant-only expressions such as
`42` are permitted with one declared argument and produce one constant result
per element. Parentheses, unary minus, division, tensor literals and general
let/function expressions are outside this new entry's grammar. The legacy
literal/scalar parser behavior is retained.

The new parser entry validates and normalizes this grammar before invoking the
existing parser and AST precedence handling. It does not expose the legacy
parser's unchecked operand indexing to malformed external-input expressions.

## Compilation boundary

`TensorArgumentSpec` carries only a name and element count. The shared
`translateTensorExpression` entry takes a span of these specs and never sees
host element values. It emits `func.func @tensor_expression` with two explicit
memref arguments: borrowed input and independent output. A single existing
`letalg.tensor_map` contains the elementwise scalar region.

Both literal translation and external-input translation use the same map-body
builder. The external path supplies output as a function argument; the literal
path retains its allocation. Both CPU APIs share context setup, the production
SCF/LLVM lowering sequence and the optimizing JIT helper. Production code does
not import the F5 raw-IR test adapter.

The external runner calls `_mlir_ciface_tensor_expression` with two rank-1
strided-memref descriptors. They describe contiguous host buffers with offset 0
and stride 1. The input descriptor ABI has mutable pointer fields, requiring a
const cast; verified map semantics only read that buffer. Generated code does
not allocate or free the caller's buffers. The only output allocation belongs
to the returned C++ vector.

## Data-driven tests

Each new source run has explicit `arguments` alongside the existing program
and assertions. Inputs are per run so one expression can be tested with
multiple values and lengths in one file:

```ocaml
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: tensor-input-cpu
    arguments:
      - name: xs
        values: [-2, 0, 3]
*)
(* @assert:cpu:result
[5, 1, 10]
*)
xs * xs + 1
```

`tensor-input-cpu` calls the production public API and checks caller data is
unchanged. `tensor-input-ir` translates through the shared frontend without
execution, supporting full structural or FileCheck IR assertions. These are
separate from the F5 `tensor-cpu-ir`/`tensor-metal-ir` raw-IR adapters and their
file-level `tensor-input` field. `tensor-input-metal` is not registered yet.

See `test/external_input/polynomial.ml` for same-expression/different-data calls,
length specialization, extreme values and two identical full expected IR
snapshots from different data. Other fixtures cover identity, unused inputs,
wrapping subtraction, precedence, whitespace, empty data and invalid syntax.
Inputs/configuration mistakes fail rather than skipping. The dedicated
`cpu_input_contracts` check reads its numeric program/data from that fixture and
additionally checks ownership, result lifetime, empty source text, stale dumps,
direct argument-count errors and synthetic size bounds.

```sh
cmake -S . -B build
cmake --build build --target tester tensor_interfaces metal_runtime_checks metal_stub_checks cpu_input_contracts -j3
./build/tester --suite test/external_input
ctest --test-dir build --output-on-failure
```

Local validation on LLVM/MLIR 22.1.8: 30 new data runs passed; CTest 7/7
passed. The complete suite passed 177 runs with Metal required on Apple M1,
and 140 runs with 37 explicit skips with Metal disabled. M2-A remains a CPU
feature; the required-device run checks that existing Metal paths did not
regress, not that external input is already supported on GPU.
