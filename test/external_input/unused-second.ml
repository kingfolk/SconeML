(* @test
version: 1
input: source
runs:
  - name: unused-second
    pipeline: tensor-input-cpu
    arguments:
      - name: "xs"
        values: [1]
      - name: "ys"
        values: [2]
  - name: gpu-unused-second
    pipeline: tensor-input-metal
    arguments:
      - name: "xs"
        values: [1]
      - name: "ys"
        values: [2]
*)
(* @assert:unused-second:result
[1]
*)
(* @assert:gpu-unused-second:result
[1]
*)(* @assert:gpu-unused-second:compare
unused-second
*)
xs
