(* @test
version: 1
input: source
runs:
  - name: rejected
    pipeline: tensor-input-cpu
    arguments:
      - name: "xs"
        values: [1]
*)
(* @assert:rejected:error
stage: frontend
message: missing an operand
*)
xs +
