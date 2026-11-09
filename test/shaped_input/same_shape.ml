(* @case rank1-square *)
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: shaped-input-cpu
    arguments:
      - name: a
        values: [1, 2, 3, 4]
        shape: [4]
*)
(* @assert:cpu:result
[2, 5, 10, 17]
*)
(* @assert:cpu:shape
[4]
*)
(* @assert:cpu:ir-check
CHECK: llvm.func @shaped_tensor_expression
CHECK-NOT: linalg.
CHECK-NOT: memref.
CHECK-NOT: scf.
CHECK: llvm.func @_mlir_ciface_shaped_tensor_expression
*)
a * a + 1

(* @case rank1-two *)
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: shaped-input-cpu
    arguments:
      - name: a
        values: [1, -2, 30]
        shape: [3]
      - name: b
        values: [4, -5, 6]
        shape: [3]
*)
(* @assert:cpu:result
[-7, 8, 18]
*)
(* @assert:cpu:shape
[3]
*)
(* @assert:cpu:ir-check
CHECK: llvm.func @shaped_tensor_expression
CHECK-NOT: linalg.
CHECK-NOT: memref.
CHECK-NOT: scf.
CHECK: llvm.func @_mlir_ciface_shaped_tensor_expression
*)
a - b * 2

(* @case rank2-one *)
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: shaped-input-cpu
    arguments:
      - name: a
        values: [1, -2, 3, 4, 5, -6]
        shape: [2, 3]
*)
(* @assert:cpu:result
[-2, 1, 6, 13, 22, 33]
*)
(* @assert:cpu:shape
[2, 3]
*)
(* @assert:cpu:ir-check
CHECK: llvm.func @shaped_tensor_expression
CHECK-NOT: linalg.
CHECK-NOT: memref.
CHECK-NOT: scf.
CHECK: llvm.func @_mlir_ciface_shaped_tensor_expression
*)
a * a - 3

(* @case rank2-two *)
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: shaped-input-cpu
    arguments:
      - name: a
        values: [1, 2, 3, 4, 5, 6]
        shape: [2, 3]
      - name: b
        values: [10, 20, 30, 40, 50, 60]
        shape: [2, 3]
  - name: ir
    pipeline: shaped-input-ir
    arguments:
      - name: a
        values: [1, 2, 3, 4, 5, 6]
        shape: [2, 3]
      - name: b
        values: [10, 20, 30, 40, 50, 60]
        shape: [2, 3]
*)
(* @assert:cpu:result
[11, 22, 33, 44, 55, 66]
*)
(* @assert:cpu:shape
[2, 3]
*)
(* @assert:cpu:ir-check
CHECK: llvm.func @shaped_tensor_expression
CHECK-NOT: linalg.
CHECK-NOT: memref.
CHECK-NOT: scf.
CHECK: llvm.func @_mlir_ciface_shaped_tensor_expression
*)
(* @assert:ir:ir-check
CHECK: affine_map<(d0, d1) -> (d0, d1)>
CHECK: func.func @shaped_tensor_expression
CHECK-SAME: memref<6xi32>
CHECK: memref.reinterpret_cast %arg0
CHECK-SAME: sizes: [2, 3], strides: [3, 1]
CHECK: memref.reinterpret_cast %arg1
CHECK: memref.reinterpret_cast %arg2
CHECK: linalg.generic
CHECK-SAME: iterator_types = ["parallel", "parallel"]
CHECK: ^bb0({{.*}}: i32, {{.*}}: i32, {{.*}}: i32):
CHECK: arith.addi
CHECK: linalg.yield
CHECK: return
*)
a + b

(* @case rank2-three *)
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: shaped-input-cpu
    arguments:
      - name: a
        values: [1, 2, 3, 4, 5, 6]
        shape: [2, 3]
      - name: b
        values: [10, 20, 30, 40, 50, 60]
        shape: [2, 3]
      - name: c
        values: [100, 200, 300, 400, 500, 600]
        shape: [2, 3]
*)
(* @assert:cpu:result
[109, 238, 387, 556, 745, 954]
*)
(* @assert:cpu:shape
[2, 3]
*)
(* @assert:cpu:ir-check
CHECK: llvm.func @shaped_tensor_expression
CHECK-NOT: linalg.
CHECK-NOT: memref.
CHECK-NOT: scf.
CHECK: llvm.func @_mlir_ciface_shaped_tensor_expression
*)
a * b + c - a

(* @case wrapping *)
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: shaped-input-cpu
    arguments:
      - name: a
        values: [2147483647, -2147483648, -1, 65536]
        shape: [2, 2]
      - name: b
        values: [2, -1, -2147483648, 65536]
        shape: [2, 2]
*)
(* @assert:cpu:result
[-1, -2147483647, -2147483647, 1]
*)
(* @assert:cpu:shape
[2, 2]
*)
(* @assert:cpu:ir-check
CHECK: llvm.func @shaped_tensor_expression
CHECK-NOT: linalg.
CHECK-NOT: memref.
CHECK-NOT: scf.
CHECK: llvm.func @_mlir_ciface_shaped_tensor_expression
*)
a * b + 1

(* @case unused-input *)
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: shaped-input-cpu
    arguments:
      - name: a
        values: [99, 88]
        shape: [2, 1]
      - name: b
        values: [3, 4]
        shape: [2, 1]
      - name: c
        values: [55, 66]
        shape: [2, 1]
*)
(* @assert:cpu:result
[4, 5]
*)
(* @assert:cpu:shape
[2, 1]
*)
(* @assert:cpu:ir-check
CHECK: llvm.func @shaped_tensor_expression
CHECK-NOT: linalg.
CHECK-NOT: memref.
CHECK-NOT: scf.
CHECK: llvm.func @_mlir_ciface_shaped_tensor_expression
*)
b + 1

(* @case constant *)
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: shaped-input-cpu
    arguments:
      - name: a
        values: [99, 88, 77]
        shape: [1, 3]
*)
(* @assert:cpu:result
[20, 20, 20]
*)
(* @assert:cpu:shape
[1, 3]
*)
(* @assert:cpu:ir-check
CHECK: llvm.func @shaped_tensor_expression
CHECK-NOT: linalg.
CHECK-NOT: memref.
CHECK-NOT: scf.
CHECK: llvm.func @_mlir_ciface_shaped_tensor_expression
*)
7 * 3 - 1

(* @case singleton *)
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: shaped-input-cpu
    arguments:
      - name: a
        values: [5]
        shape: [1, 1]
*)
(* @assert:cpu:result
[6]
*)
(* @assert:cpu:shape
[1, 1]
*)
(* @assert:cpu:ir-check
CHECK: llvm.func @shaped_tensor_expression
CHECK-NOT: linalg.
CHECK-NOT: memref.
CHECK-NOT: scf.
CHECK: llvm.func @_mlir_ciface_shaped_tensor_expression
*)
a + 1

(* @case empty-rank1 *)
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: shaped-input-cpu
    arguments:
      - name: a
        values: []
        shape: [0]
*)
(* @assert:cpu:result
[]
*)
(* @assert:cpu:shape
[0]
*)
a + 1

(* @case empty-rows *)
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: shaped-input-cpu
    arguments:
      - name: a
        values: []
        shape: [0, 3]
      - name: b
        values: []
        shape: [0, 3]
*)
(* @assert:cpu:result
[]
*)
(* @assert:cpu:shape
[0, 3]
*)
a + b

(* @case empty-columns *)
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: shaped-input-cpu
    arguments:
      - name: a
        values: []
        shape: [2, 0]
*)
(* @assert:cpu:result
[]
*)
(* @assert:cpu:shape
[2, 0]
*)
7

(* @case alias *)
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: shaped-input-cpu
    arguments:
      - name: a
        values: [1, 2, 3, 4, 5, 6]
        shape: [2, 3]
      - name: b
        values: [1, 2, 3, 4, 5, 6]
        shape: [2, 3]
      - name: c
        values: [100, 200, 300, 400, 500, 600]
        shape: [2, 3]
*)
(* @assert:cpu:result
[101, 204, 309, 416, 525, 636]
*)
(* @assert:cpu:shape
[2, 3]
*)
(* @assert:cpu:ir-check
CHECK: llvm.func @shaped_tensor_expression
CHECK-NOT: linalg.
CHECK-NOT: memref.
CHECK-NOT: scf.
CHECK: llvm.func @_mlir_ciface_shaped_tensor_expression
*)
a * b + c

(* @case overlap *)
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: shaped-input-cpu
    arguments:
      - name: a
        values: [1, 2, 3, 4, 5, 6]
        shape: [2, 3]
      - name: b
        values: [3, 4, 5, 6, 7, 8]
        shape: [2, 3]
      - name: c
        values: [10, 20, 30, 40, 50, 60]
        shape: [2, 3]
*)
(* @assert:cpu:result
[13, 28, 45, 64, 85, 108]
*)
(* @assert:cpu:shape
[2, 3]
*)
(* @assert:cpu:ir-check
CHECK: llvm.func @shaped_tensor_expression
CHECK-NOT: linalg.
CHECK-NOT: memref.
CHECK-NOT: scf.
CHECK: llvm.func @_mlir_ciface_shaped_tensor_expression
*)
a * b + c
