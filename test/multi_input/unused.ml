(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: tensor-input-cpu
    arguments:
      - name: "a"
        values: [1, 2, 3]
      - name: "unused"
        values: [10, 20, 30]
      - name: "also_unused"
        values: [4, 5, 6]
  - name: metal
    pipeline: tensor-input-metal
    threads-per-group: 4
    arguments:
      - name: "a"
        values: [1, 2, 3]
      - name: "unused"
        values: [10, 20, 30]
      - name: "also_unused"
        values: [4, 5, 6]
*)
(* @assert:cpu:result
[2, 3, 4]
*)
(* @assert:metal:result
[2, 3, 4]
*)
(* @assert:metal:compare
cpu
*)
a + 1
