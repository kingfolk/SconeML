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
[1, 4, 9, 16]
*)
(* @assert:metal:result
[1, 4, 9, 16]
*)
(* @assert:metal:compare
cpu
*)
let xs = [0, 1, 2, 3] in xs * xs + 2 * xs + 1
