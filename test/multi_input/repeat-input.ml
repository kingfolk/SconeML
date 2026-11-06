(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: tensor-input-cpu
    arguments:
      - name: "a"
        values: [1, 2, 3]
      - name: "b"
        values: [10, 20, 30]
      - name: "c"
        values: [4, 5, 6]
  - name: metal
    pipeline: tensor-input-metal
    threads-per-group: 4
    arguments:
      - name: "a"
        values: [1, 2, 3]
      - name: "b"
        values: [10, 20, 30]
      - name: "c"
        values: [4, 5, 6]
*)
(* @assert:cpu:result
[-39, -96, -171]
*)
(* @assert:metal:result
[-39, -96, -171]
*)
(* @assert:metal:compare
cpu
*)
a * a - b * c
