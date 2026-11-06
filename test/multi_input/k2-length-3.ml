(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: tensor-input-cpu
    arguments:
      - name: "a"
        values: [-2147483648, 506952121, -1133579406]
      - name: "b"
        values: [2147483647, 99338858, -1948805931]
  - name: metal
    pipeline: tensor-input-metal
    threads-per-group: 4
    arguments:
      - name: "a"
        values: [-2147483648, 506952121, -1133579406]
      - name: "b"
        values: [2147483647, 99338858, -1948805931]
*)
(* @assert:cpu:result
[0, -2069895455, -1956806040]
*)
(* @assert:metal:result
[0, -2069895455, -1956806040]
*)
(* @assert:metal:compare
cpu
*)
a * b - a
