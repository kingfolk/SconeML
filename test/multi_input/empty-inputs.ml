(* @case two-empty *)
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: tensor-input-cpu
    arguments:
      - name: "a"
        values: []
      - name: "b"
        values: []
  - name: metal
    pipeline: tensor-input-metal
    threads-per-group: 4
    arguments:
      - name: "a"
        values: []
      - name: "b"
        values: []
*)
(* @assert:cpu:result
[]
*)
(* @assert:metal:result
[]
*)
(* @assert:metal:compare
cpu
*)
a - b

(* @case three-empty *)
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: tensor-input-cpu
    arguments:
      - name: "a"
        values: []
      - name: "b"
        values: []
      - name: "c"
        values: []
  - name: metal
    pipeline: tensor-input-metal
    threads-per-group: 4
    arguments:
      - name: "a"
        values: []
      - name: "b"
        values: []
      - name: "c"
        values: []
*)
(* @assert:cpu:result
[]
*)
(* @assert:metal:result
[]
*)
(* @assert:metal:compare
cpu
*)
a * b + c
