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
      - name: "c"
        values: [-7, -1028477394, -2056954781]
  - name: metal
    pipeline: tensor-input-metal
    threads-per-group: 4
    arguments:
      - name: "a"
        values: [-2147483648, 506952121, -1133579406]
      - name: "b"
        values: [2147483647, 99338858, -1948805931]
      - name: "c"
        values: [-7, -1028477394, -2056954781]
*)
(* @assert:cpu:result
[2147483641, 2109209765, -1634368411]
*)
(* @assert:metal:result
[2147483641, 2109209765, -1634368411]
*)
(* @assert:metal:compare
cpu
*)
a * a - b * c
