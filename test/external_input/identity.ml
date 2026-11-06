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
  - name: gpu-identity
    pipeline: tensor-input-metal
    threads-per-group: 4
    arguments:
      - name: "Input_2"
        values: [-2147483648, 2147483647, -1, 0, 1, 46341]
  - name: gpu-empty
    pipeline: tensor-input-metal
    threads-per-group: 4
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
(* @assert:gpu-identity:result
[-2147483648, 2147483647, -1, 0, 1, 46341]
*)
(* @assert:gpu-identity:compare
identity
*)
(* @assert:gpu-empty:result
[]
*)
(* @assert:gpu-empty:compare
empty
*)
Input_2
