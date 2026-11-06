(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: tensor-input-cpu
    arguments:
      - name: "a"
        values: [-2147483648, 506952121, -1133579406, 1520856363, -119675164]
      - name: "b"
        values: [2147483647, 99338858, -1948805931, 298016576, -1750128213]
  - name: metal
    pipeline: tensor-input-metal
    threads-per-group: 4
    arguments:
      - name: "a"
        values: [-2147483648, 506952121, -1133579406, 1520856363, -119675164]
      - name: "b"
        values: [2147483647, 99338858, -1948805931, 298016576, -1750128213]
*)
(* @assert:cpu:result
[0, -2069895455, -1956806040, 339268245, 523360104]
*)
(* @assert:metal:result
[0, -2069895455, -1956806040, 339268245, 523360104]
*)
(* @assert:metal:compare
cpu
*)
a * b - a
