# External tensors on CPU and Metal

`runTensorCPU` and `runTensorGPU` execute an elementwise expression over
caller-provided host data. Both accept one to three named, contiguous, equal-length i32 inputs
and return a new owning host vector. M2-C extends the original M2-A/B
single-input path; see [multi-input maps](multi-input-tensor-map.md). The existing literal CPU/GPU APIs remain available.

```cpp
#include "src/tensor/CPURunner.h"
#include "src/tensor/GPURunner.h"
#include <vector>

std::vector<int32_t> values{-2, 0, 3};
std::vector<sconeml::tensor::TensorArgument> arguments{{"xs", values}};
auto result = sconeml::tensor::runTensorCPU("xs * xs + 1", arguments);
auto gpu = sconeml::tensor::runTensorGPU("xs * xs + 1", arguments);
// Both return {5, 1, 10}; values is unchanged. Nonempty GPU calls require Metal.
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
std::vector<int32_t> runTensorGPU(
    const std::string &expression,
    std::span<const TensorArgument> arguments,
    std::string *gpuSource = nullptr, MetalLaunchOptions options = {});
```

The input data is borrowed until the synchronous call returns. Do not destroy
or concurrently modify its storage during the call. The runner does not free
or write input data. Output is allocated by the runner and remains valid after
both the input storage and JIT engine are destroyed. Modifying output does not
modify input. The named wrapper owns its name, not its tensor storage.

Every nonempty CPU call translates, lowers and creates an LLVM JIT independently.
Every nonempty GPU call translates, lowers to GPU IR, exports MSL and invokes
the one-shot Metal runtime. Input is copied into a Metal buffer; CPU execution
borrows its host address directly. No path modifies or takes ownership of input.
Calling the same expression with different values or lengths is supported;
there is no compile-once object or cache. The function's memref shape is
specialized to the input length for each call, not represented as dynamic IR.

For zero elements, both runners still validate the name, expression and map.
The CPU runner returns without creating a JIT; its optional LLVM dump is cleared
and remains empty. The GPU runner also lowers/exports and returns the generated
MSL in its optional dump, then takes the runtime no-op without device lookup,
shader compilation, Metal buffer allocation or dispatch. Invalid launch widths
are rejected even for zero count. A misspelled name or malformed
expression still fails for empty data. Size/index/byte overflow is rejected;
allocation failures propagate normally for otherwise representable large data.

## Expression syntax

The initial grammar is deliberately restricted:

```text
operand (('+' | '-' | '*') operand)*
operand = declared-argument-name | nonnegative-i32-constant
```

Multiplication binds before addition/subtraction; addition and subtraction
associate left to right. ASCII argument names contain letters, digits and
underscore, cannot begin with a digit, and cannot be reserved names such as
`let` or `if`. Spaces, tabs and newlines are accepted. Unknown variables,
missing operands/operators, unsupported characters and out-of-range constants
raise an exception before IR construction or JIT execution.

Negative/extreme host elements are supported independently of constant syntax.
Operators preserve modulo-2^32 i32 semantics. Constant-only expressions such as
`42` are permitted with declared arguments and produce one constant result
per element. Parentheses, unary minus, division, tensor literals and general
let/function expressions are outside this new entry's grammar. The legacy
literal/scalar parser behavior is retained.

The new parser entry validates and normalizes this grammar before invoking the
existing parser and AST precedence handling. It does not expose the legacy
parser's unchecked operand indexing to malformed external-input expressions.

## Compilation boundary

`TensorArgumentSpec` carries only a name and element count. The shared
`translateTensorExpression` entry takes a span of these specs and never sees
host element values. It emits `func.func @tensor_expression` with K+1 explicit
memref arguments: K borrowed inputs and an independent output. A single existing
`letalg.tensor_map` contains the elementwise scalar region.

Both literal translation and external-input translation use the same map-body
builder. The external path supplies output as a function argument; the literal
path retains its allocation. Both CPU APIs share context setup, the production
SCF/LLVM lowering sequence and the optimizing JIT helper. Both GPU APIs share
context setup and map lowering/export/dispatch; only the literal API scans
alloc/store operations to recover literal data. External GPU execution directly
passes the caller span to the existing runtime. Production code does
not import the F5 raw-IR test adapter.

The external runner calls `_mlir_ciface_tensor_expression` with K+1 rank-1
strided-memref descriptors, using an exact function-pointer signature for K=1/2/3. They describe contiguous host buffers with offset 0
and stride 1. The input descriptor ABI has mutable pointer fields, requiring a
const cast; verified map semantics only read that buffer. Generated code does
not allocate or free the caller's buffers. The only output allocation belongs
to the returned C++ vector.

## Data-driven tests

Each new source run has explicit `arguments` alongside the existing program
and assertions. Inputs are per run so one expression can be tested with
multiple values and lengths in one file (replace the pipeline with
`tensor-input-metal` to run the same input on GPU):

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
file-level `tensor-input` field. `tensor-input-metal` calls the production external GPU API, returning numeric
results and the actually exported/executed MSL. It accepts `threads-per-group`,
plus numeric, compare, text and success/error assertions.

See `test/external_input/polynomial.ml` for same-expression/different-data calls,
length specialization, extreme values and two identical full expected IR
snapshots from different data. Other fixtures cover identity, unused inputs,
wrapping subtraction, precedence, whitespace, empty data and invalid syntax.
Frontend and host launch-bound errors are checked before device skips.
Nonempty GPU runs explicitly skip when Metal is unavailable or `--no-metal` is
selected; `--require-metal` turns those skips into failures. Empty GPU calls and
invalid frontend/host-launch tests still run without a device. The dedicated
`cpu_input_contracts` check reads its numeric program/data from that fixture and
additionally checks ownership, result lifetime, empty source text, stale dumps,
direct argument-count errors and synthetic size bounds.

```sh
cmake -S . -B build
cmake --build build --target tester tensor_interfaces metal_runtime_checks metal_stub_checks cpu_input_contracts gpu_input_contracts gpu_input_stub_contracts -j3
./build/tester --suite test/external_input
ctest --test-dir build --output-on-failure
```

M2-A baseline, before GPU integration: 30 new data runs and CTest 7/7 passed
on LLVM/MLIR 22.1.8. The suite passed 177 runs with Metal required on Apple M1,
and 140 runs with 37 skips with Metal disabled. Those baseline counts verified
the CPU feature and old Metal regressions; M2-B results below additionally
verify actual external-input GPU execution.

## External-input GPU execution

Use `runTensorGPU(expression, arguments, &source, {4})` to request four threads
per group; omit options for the runtime default. The existing input/output/count
bindings (0/1/2), separate output allocation, uniform padded groups and u32
launch bounds are unchanged. The original M2-B API added data binding and execution through
the shared frontend without another scalar compiler or launch algorithm.
A nonempty GPU call without a device throws and never invokes the CPU runner.
The non-Apple stub has the same validated empty behavior and rejects nonempty
execution. M2-B introduced no new source grammar or cache; M2-C multi-input support is described below.

The existing source cases now include GPU runs with explicit results and CPU
comparisons. `test/external_input/metal` adds mixed wrapping arithmetic at
lengths 1/4/5/31/32/33/255/256/257/513, explicit widths 1/4/32/256 and automatic
widths, plus invalid launch widths for empty/nonempty arrays. Expected arrays
are computed independently with Python integer arithmetic reduced to signed
32-bit patterns. They are stored in the same files as program/configuration;
tests do not derive expectations from backend output.

`gpu_input_contracts` directly checks the public API's empty/validation behavior,
no-device rejection, input preservation, independent output, results surviving
input/device-resource destruction, repeated calls and data-independent MSL.
`gpu_input_stub_contracts` links the actual stub even on macOS and exercises the
same public GPU API. These targets do not link the CPU runner/ExecutionEngine.

Local M2-B validation on Apple M1 / LLVM+MLIR 22.1.8: external-input matrix
90 PASS / 0 SKIP / 0 FAIL; full strict Metal suite 237 PASS / 0 SKIP / 0 FAIL;
forced no-device suite 172 PASS / 65 SKIP / 0 FAIL. CTest has nine entries.
Direct API GPU ownership/execution contracts also passed with real Metal access.
The explicit stub is tested on this Mac, not a separate Linux platform build.

M2-C now generalizes the same APIs to 1-3 inputs. All names must be unique and
all lengths must match, including unused arguments and empty arrays. Read-only
host spans may alias; output remains independent. K=1 shader text/bindings and
literal convenience behavior are preserved. For K>1 inputs bind at 0..K-1,
output at K and count at K+1; the exported inputCount is checked by the runtime.
The original single-input runMetalKernel overload delegates to the generalized
implementation. See [M2-C details and tests](multi-input-tensor-map.md).
