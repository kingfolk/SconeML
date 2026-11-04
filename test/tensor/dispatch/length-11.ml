(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: tensor-cpu
  - name: metal
    pipeline: tensor-metal
    threads-per-group: 4
*)
(* @assert:cpu:result
[1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121]
*)
(* @assert:metal:result
[1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121]
*)
(* @assert:metal:compare
cpu
*)
let xs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] in xs * xs + 2 * xs + 1
