(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: tensor-input-cpu
    arguments:
      - name: "b"
        values: [10, 20, 30]
      - name: "c"
        values: [4, 5, 6]
      - name: "a"
        values: [1, 2, 3]
  - name: metal
    pipeline: tensor-input-metal
    threads-per-group: 4
    arguments:
      - name: "b"
        values: [10, 20, 30]
      - name: "c"
        values: [4, 5, 6]
      - name: "a"
        values: [1, 2, 3]
*)
(* @assert:cpu:result
[-6, -35, -84]
*)
(* @assert:metal:result
[-6, -35, -84]
*)
(* @assert:metal:compare
cpu
*)
c - a * b
