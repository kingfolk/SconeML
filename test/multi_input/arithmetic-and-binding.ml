(* @case two-add *)
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

(* @case three-fma *)
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
      - name: "c"
        values: [4, 5, 6]
  - name: metal
    pipeline: tensor-input-metal
    threads-per-group: 4
    arguments:
      - name: "a"
        values: [1, 2, 3]
      - name: "b"
        values: [10, 20, 30]
      - name: "c"
        values: [4, 5, 6]
*)
(* @assert:cpu:result
[14, 45, 96]
*)
(* @assert:metal:result
[14, 45, 96]
*)
(* @assert:metal:compare
cpu
*)
a * b + c

(* @case binding-order *)
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: tensor-input-cpu
    arguments:
      - name: "b"
        values: [10, 20, 30]
      - name: "c"
        values: [4, 5, 6]
      - name: "a"
        values: [1, 2, 3]
  - name: metal
    pipeline: tensor-input-metal
    threads-per-group: 4
    arguments:
      - name: "b"
        values: [10, 20, 30]
      - name: "c"
        values: [4, 5, 6]
      - name: "a"
        values: [1, 2, 3]
*)
(* @assert:cpu:result
[-6, -35, -84]
*)
(* @assert:metal:result
[-6, -35, -84]
*)
(* @assert:metal:compare
cpu
*)
c - a * b

(* @case repeat-input *)
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
      - name: "c"
        values: [4, 5, 6]
  - name: metal
    pipeline: tensor-input-metal
    threads-per-group: 4
    arguments:
      - name: "a"
        values: [1, 2, 3]
      - name: "b"
        values: [10, 20, 30]
      - name: "c"
        values: [4, 5, 6]
*)
(* @assert:cpu:result
[-39, -96, -171]
*)
(* @assert:metal:result
[-39, -96, -171]
*)
(* @assert:metal:compare
cpu
*)
a * a - b * c

(* @case unused *)
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: tensor-input-cpu
    arguments:
      - name: "a"
        values: [1, 2, 3]
      - name: "unused"
        values: [10, 20, 30]
      - name: "also_unused"
        values: [4, 5, 6]
  - name: metal
    pipeline: tensor-input-metal
    threads-per-group: 4
    arguments:
      - name: "a"
        values: [1, 2, 3]
      - name: "unused"
        values: [10, 20, 30]
      - name: "also_unused"
        values: [4, 5, 6]
*)
(* @assert:cpu:result
[2, 3, 4]
*)
(* @assert:metal:result
[2, 3, 4]
*)
(* @assert:metal:compare
cpu
*)
a + 1

(* @case constant *)
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
      - name: "c"
        values: [4, 5, 6]
  - name: metal
    pipeline: tensor-input-metal
    threads-per-group: 4
    arguments:
      - name: "a"
        values: [1, 2, 3]
      - name: "b"
        values: [10, 20, 30]
      - name: "c"
        values: [4, 5, 6]
*)
(* @assert:cpu:result
[7, 7, 7]
*)
(* @assert:metal:result
[7, 7, 7]
*)
(* @assert:metal:compare
cpu
*)
7
