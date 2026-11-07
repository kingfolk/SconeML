(* @case division *)
(* @test
version: 1
input: source
runs:
  - name: rejected
    pipeline: tensor-input-cpu
    arguments:
      - name: "xs"
        values: [1]
  - name: gpu-rejected
    pipeline: tensor-input-metal
    arguments:
      - name: "xs"
        values: [1]
*)
(* @assert:rejected:error
stage: frontend
message: expected +, - or *
*)
(* @assert:gpu-rejected:error
stage: frontend
message: expected +, - or *
*)
xs / 2

(* @case double-operator *)
(* @test
version: 1
input: source
runs:
  - name: rejected
    pipeline: tensor-input-cpu
    arguments:
      - name: "xs"
        values: [1]
  - name: gpu-rejected
    pipeline: tensor-input-metal
    arguments:
      - name: "xs"
        values: [1]
*)
(* @assert:rejected:error
stage: frontend
message: expected an identifier
*)
(* @assert:gpu-rejected:error
stage: frontend
message: expected an identifier
*)
xs + * 1

(* @case grouping *)
(* @test
version: 1
input: source
runs:
  - name: rejected
    pipeline: tensor-input-cpu
    arguments:
      - name: "xs"
        values: [1]
  - name: gpu-rejected
    pipeline: tensor-input-metal
    arguments:
      - name: "xs"
        values: [1]
*)
(* @assert:rejected:error
stage: frontend
message: expected an identifier
*)
(* @assert:gpu-rejected:error
stage: frontend
message: expected an identifier
*)
(xs + 1)

(* @case leading-operator *)
(* @test
version: 1
input: source
runs:
  - name: rejected
    pipeline: tensor-input-cpu
    arguments:
      - name: "xs"
        values: [1]
  - name: gpu-rejected
    pipeline: tensor-input-metal
    arguments:
      - name: "xs"
        values: [1]
*)
(* @assert:rejected:error
stage: frontend
message: expected an identifier
*)
(* @assert:gpu-rejected:error
stage: frontend
message: expected an identifier
*)
* xs

(* @case let *)
(* @test
version: 1
input: source
runs:
  - name: rejected
    pipeline: tensor-input-cpu
    arguments:
      - name: "xs"
        values: [1]
  - name: gpu-rejected
    pipeline: tensor-input-metal
    arguments:
      - name: "xs"
        values: [1]
*)
(* @assert:rejected:error
stage: frontend
message: expected an identifier
*)
(* @assert:gpu-rejected:error
stage: frontend
message: expected an identifier
*)
let xs = 1 in xs

(* @case literal *)
(* @test
version: 1
input: source
runs:
  - name: rejected
    pipeline: tensor-input-cpu
    arguments:
      - name: "xs"
        values: [1]
  - name: gpu-rejected
    pipeline: tensor-input-metal
    arguments:
      - name: "xs"
        values: [1]
*)
(* @assert:rejected:error
stage: frontend
message: expected an identifier
*)
(* @assert:gpu-rejected:error
stage: frontend
message: expected an identifier
*)
[1, 2]

(* @case missing-operator *)
(* @test
version: 1
input: source
runs:
  - name: rejected
    pipeline: tensor-input-cpu
    arguments:
      - name: "xs"
        values: [1]
  - name: gpu-rejected
    pipeline: tensor-input-metal
    arguments:
      - name: "xs"
        values: [1]
*)
(* @assert:rejected:error
stage: frontend
message: expected +, - or *
*)
(* @assert:gpu-rejected:error
stage: frontend
message: expected +, - or *
*)
xs 1

(* @case trailing-operator *)
(* @test
version: 1
input: source
runs:
  - name: rejected
    pipeline: tensor-input-cpu
    arguments:
      - name: "xs"
        values: [1]
  - name: gpu-rejected
    pipeline: tensor-input-metal
    arguments:
      - name: "xs"
        values: [1]
*)
(* @assert:rejected:error
stage: frontend
message: missing an operand
*)
(* @assert:gpu-rejected:error
stage: frontend
message: missing an operand
*)
xs +

(* @case unary-minus *)
(* @test
version: 1
input: source
runs:
  - name: rejected
    pipeline: tensor-input-cpu
    arguments:
      - name: "xs"
        values: [1]
  - name: gpu-rejected
    pipeline: tensor-input-metal
    arguments:
      - name: "xs"
        values: [1]
*)
(* @assert:rejected:error
stage: frontend
message: expected an identifier
*)
(* @assert:gpu-rejected:error
stage: frontend
message: expected an identifier
*)
-1

(* @case wide-constant *)
(* @test
version: 1
input: source
runs:
  - name: rejected
    pipeline: tensor-input-cpu
    arguments:
      - name: "xs"
        values: [1]
  - name: gpu-rejected
    pipeline: tensor-input-metal
    arguments:
      - name: "xs"
        values: [1]
*)
(* @assert:rejected:error
stage: frontend
message: constant outside i32 range
*)
(* @assert:gpu-rejected:error
stage: frontend
message: constant outside i32 range
*)
xs + 2147483648
