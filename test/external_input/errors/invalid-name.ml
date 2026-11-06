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
