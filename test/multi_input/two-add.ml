(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: tensor-input-cpu
    arguments:
      - name: "a"
        values: [1, 2, 3]
      - name: "b"
        values: [10, 20, 30]
  - name: metal
    pipeline: tensor-input-metal
    threads-per-group: 4
    arguments:
      - name: "a"
        values: [1, 2, 3]
      - name: "b"
        values: [10, 20, 30]
*)
(* @assert:cpu:result
[11, 22, 33]
*)
(* @assert:metal:result
[11, 22, 33]
*)
(* @assert:metal:compare
cpu
*)
(* @assert:metal:text-check
CHECK: device const uint *input0
CHECK-SAME: buffer(0)
CHECK: device const uint *input1
CHECK-SAME: buffer(1)
CHECK: device uint *output
CHECK-SAME: buffer(2)
CHECK: constant uint &count
CHECK-SAME: buffer(3)
*)
a + b
