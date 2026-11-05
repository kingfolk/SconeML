(* @test
version: 1
input: source
runs:
  - name: rejected
    pipeline: tensor-input-cpu
    arguments: []
*)
(* @assert:rejected:error
stage: frontend
message: requires exactly one tensor argument
*)
xs
