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
  - name: gpu-constant
    pipeline: tensor-input-metal
    threads-per-group: 4
    arguments:
      - name: "unused"
        values: [-9, 0, 2147483647]
  - name: gpu-empty
    pipeline: tensor-input-metal
    threads-per-group: 4
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
(* @assert:gpu-constant:result
[42, 42, 42]
*)
(* @assert:gpu-constant:compare
constant
*)
(* @assert:gpu-empty:result
[]
*)
(* @assert:gpu-empty:compare
empty
*)
42
