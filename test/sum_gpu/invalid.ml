(* @case negative-axis *)
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: sum-i32-metal
    arguments:
      - name: a
        values: []
        shape: [0, 3]
*)
(* @assert:cpu:error
stage: frontend
message: "nonnegative integer"
*)
sum(a, axis=-1)

(* @case axis-rank *)
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: sum-i32-metal
    arguments:
      - name: a
        values: []
        shape: [0, 3]
*)
(* @assert:cpu:error
stage: frontend
message: "outside input rank"
*)
sum(a, axis=2)

(* @case axis-list *)
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: sum-i32-metal
    arguments:
      - name: a
        values: []
        shape: [0, 3]
*)
(* @assert:cpu:error
stage: frontend
message: "nonnegative integer"
*)
sum(a, axis=[0,1])

(* @case duplicate-axis *)
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: sum-i32-metal
    arguments:
      - name: a
        values: []
        shape: [0, 3]
*)
(* @assert:cpu:error
stage: frontend
message: "nonnegative integer"
*)
sum(a, axis=0, axis=1)

(* @case keepdims *)
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: sum-i32-metal
    arguments:
      - name: a
        values: []
        shape: [0, 3]
*)
(* @assert:cpu:error
stage: frontend
message: "keepdims is unsupported"
*)
sum(a, keepdims=true)

(* @case missing-operand *)
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: sum-i32-metal
    arguments:
      - name: a
        values: []
        shape: [0, 3]
*)
(* @assert:cpu:error
stage: frontend
message: "requires an operand"
*)
sum()

(* @case not-sum *)
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: sum-i32-metal
    arguments:
      - name: a
        values: []
        shape: [0, 3]
*)
(* @assert:cpu:error
stage: frontend
message: "requires sum("
*)
a

(* @case unknown *)
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: sum-i32-metal
    arguments:
      - name: a
        values: []
        shape: [0, 3]
*)
(* @assert:cpu:error
stage: frontend
message: "currently requires one named input"
*)
sum(b)

