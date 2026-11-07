(* @case length-3 *)
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
[1, 4, 9]
*)
(* @assert:metal:result
[1, 4, 9]
*)
(* @assert:metal:compare
cpu
*)
let xs = [0, 1, 2] in xs * xs + 2 * xs + 1

(* @case length-4 *)
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

(* @case length-5 *)
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
[1, 4, 9, 16, 25]
*)
(* @assert:metal:result
[1, 4, 9, 16, 25]
*)
(* @assert:metal:compare
cpu
*)
let xs = [0, 1, 2, 3, 4] in xs * xs + 2 * xs + 1

(* @case length-11 *)
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
