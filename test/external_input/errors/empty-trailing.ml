(* @test
version: 1
input: source
runs:
  - name: rejected
    pipeline: tensor-input-cpu
    arguments:
      - name: "xs"
        values: []
  - name: gpu-rejected
    pipeline: tensor-input-metal
    arguments:
      - name: "xs"
        values: []
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
