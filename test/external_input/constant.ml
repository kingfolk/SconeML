(* @test
version: 1
input: source
runs:
  - name: constant
    pipeline: tensor-input-cpu
    arguments:
      - name: "unused"
        values: [-9, 0, 2147483647]
  - name: empty
    pipeline: tensor-input-cpu
    arguments:
      - name: "unused"
        values: []
*)
(* @assert:constant:result
[42, 42, 42]
*)
(* @assert:empty:result
[]
*)
42
