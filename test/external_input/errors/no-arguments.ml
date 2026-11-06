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
