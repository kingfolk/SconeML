(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: tensor-input-cpu
    arguments:
      - name: "a"
        values: [-2147483648, 506952121, -1133579406, 1520856363]
      - name: "b"
        values: [2147483647, 99338858, -1948805931, 298016576]
      - name: "c"
        values: [-7, -1028477394, -2056954781, 1209535128]
  - name: metal
    pipeline: tensor-input-metal
    threads-per-group: 4
    arguments:
      - name: "a"
        values: [-2147483648, 506952121, -1133579406, 1520856363]
      - name: "b"
        values: [2147483647, 99338858, -1948805931, 298016576]
      - name: "c"
        values: [-7, -1028477394, -2056954781, 1209535128]
*)
(* @assert:cpu:result
[2147483641, 2109209765, -1634368411, -493316295]
*)
(* @assert:metal:result
[2147483641, 2109209765, -1634368411, -493316295]
*)
(* @assert:metal:compare
cpu
*)
a * a - b * c
