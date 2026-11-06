(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: tensor-input-cpu
    arguments:
      - name: "a"
        values: [-2, 0, 3, -2147483648, 2147483647]
      - name: "b"
        values: [-2, 0, 3, -2147483648, 2147483647]
      - name: "c"
        values: [7, 11, -3, 1, 9]
  - name: metal
    pipeline: tensor-input-metal
    threads-per-group: 4
    arguments:
      - name: "a"
        values: [-2, 0, 3, -2147483648, 2147483647]
      - name: "b"
        values: [-2, 0, 3, -2147483648, 2147483647]
      - name: "c"
        values: [7, 11, -3, 1, 9]
*)
(* @assert:cpu:result
[18, 0, 18, -2147483648, -2147483638]
*)
(* @assert:metal:result
[18, 0, 18, -2147483648, -2147483638]
*)
(* @assert:metal:compare
cpu
*)
a * a - b * c
