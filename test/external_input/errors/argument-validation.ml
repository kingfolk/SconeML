(* @case empty-name *)
(* @test
version: 1
input: source
runs:
  - name: rejected
    pipeline: tensor-input-cpu
    arguments:
      - name: ""
        values: [1]
  - name: gpu-rejected
    pipeline: tensor-input-metal
    arguments:
      - name: ""
        values: [1]
*)
(* @assert:rejected:error
stage: frontend
message: invalid tensor argument name
*)
(* @assert:gpu-rejected:error
stage: frontend
message: invalid tensor argument name
*)
xs

(* @case invalid-name *)
(* @test
version: 1
input: source
runs:
  - name: rejected
    pipeline: tensor-input-cpu
    arguments:
      - name: "2xs"
        values: [1]
  - name: gpu-rejected
    pipeline: tensor-input-metal
    arguments:
      - name: "2xs"
        values: [1]
*)
(* @assert:rejected:error
stage: frontend
message: invalid tensor argument name
*)
(* @assert:gpu-rejected:error
stage: frontend
message: invalid tensor argument name
*)
xs

(* @case no-arguments *)
(* @test
version: 1
input: source
runs:
  - name: rejected
    pipeline: tensor-input-cpu
    arguments: []
  - name: gpu-rejected
    pipeline: tensor-input-metal
    arguments: []
*)
(* @assert:rejected:error
stage: frontend
message: requires between one and three tensor arguments
*)
(* @assert:gpu-rejected:error
stage: frontend
message: requires between one and three tensor arguments
*)
xs

(* @case reserved-name *)
(* @test
version: 1
input: source
runs:
  - name: rejected
    pipeline: tensor-input-cpu
    arguments:
      - name: "let"
        values: [1]
  - name: gpu-rejected
    pipeline: tensor-input-metal
    arguments:
      - name: "let"
        values: [1]
*)
(* @assert:rejected:error
stage: frontend
message: invalid tensor argument name
*)
(* @assert:gpu-rejected:error
stage: frontend
message: invalid tensor argument name
*)
xs

(* @case unknown-name *)
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
message: "unknown tensor expression variable: ys"
*)
(* @assert:gpu-rejected:error
stage: frontend
message: "unknown tensor expression variable: ys"
*)
ys + 1
