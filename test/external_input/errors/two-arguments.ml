(* @test
version: 1
input: source
runs:
  - name: rejected
    pipeline: tensor-input-cpu
    arguments:
      - name: "xs"
        values: [1]
      - name: "ys"
        values: [2]
*)
(* @assert:rejected:error
stage: frontend
message: requires exactly one tensor argument
*)
xs
