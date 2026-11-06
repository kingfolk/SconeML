(* @test
version: 1
input: source
runs:
  - name: subtract
    pipeline: tensor-input-cpu
    arguments:
      - name: "xs"
        values: [-2147483648, 2147483647, -1, 0, 1, 46341]
  - name: gpu-subtract
    pipeline: tensor-input-metal
    threads-per-group: 4
    arguments:
      - name: "xs"
        values: [-2147483648, 2147483647, -1, 0, 1, 46341]
*)
(* @assert:subtract:result
[2147483647, -2147483648, 0, -1, -2, -46342]
*)
(* @assert:gpu-subtract:result
[2147483647, -2147483648, 0, -1, -2, -46342]
*)
(* @assert:gpu-subtract:compare
subtract
*)
0 - xs - 1
