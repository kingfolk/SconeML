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
