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
message: "unknown tensor expression variable: ys"
*)
ys + 1
