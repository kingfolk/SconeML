(* @test
version: 1
input: source
runs:
  - name: identity
    pipeline: tensor-input-cpu
    arguments:
      - name: "Input_2"
        values: [-2147483648, 2147483647, -1, 0, 1, 46341]
  - name: empty
    pipeline: tensor-input-cpu
    arguments:
      - name: "Input_2"
        values: []
*)
(* @assert:identity:result
[-2147483648, 2147483647, -1, 0, 1, 46341]
*)
(* @assert:empty:result
[]
*)
Input_2
