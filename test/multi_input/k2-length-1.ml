(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: tensor-input-cpu
    arguments:
      - name: "a"
        values: [-2147483648]
      - name: "b"
        values: [2147483647]
  - name: metal
    pipeline: tensor-input-metal
    threads-per-group: 1
    arguments:
      - name: "a"
        values: [-2147483648]
      - name: "b"
        values: [2147483647]
*)
(* @assert:cpu:result
[0]
*)
(* @assert:metal:result
[0]
*)
(* @assert:metal:compare
cpu
*)
a * b - a
