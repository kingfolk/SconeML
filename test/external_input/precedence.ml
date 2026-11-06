(* @test
version: 1
input: source
runs:
  - name: precedence
    pipeline: tensor-input-cpu
    arguments:
      - name: "xs"
        values: [-2147483648, 2147483647, -1, 0, 1, 46341]
  - name: gpu-precedence
    pipeline: tensor-input-metal
    threads-per-group: 4
    arguments:
      - name: "xs"
        values: [-2147483648, 2147483647, -1, 0, 1, 46341]
*)
(* @assert:precedence:result
[0, -2, -2, 0, 2, 92682]
*)
(* @assert:gpu-precedence:result
[0, -2, -2, 0, 2, 92682]
*)
(* @assert:gpu-precedence:compare
precedence
*)
xs + 3 * xs - 2 * xs
