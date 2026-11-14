(* @case global *)
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: sum-i32-cpu
    arguments:
      - name: a
        dtype: i32
        shape: [6]
        values: [1, 2, 3, 4, 5, 6]
*)
(* @assert:cpu:result
[21]
*)
(* @assert:cpu:shape
[]
*)
sum(a)

(* @case rows *)
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: sum-i32-cpu
    arguments:
      - name: a
        dtype: i32
        shape: [2, 3]
        values: [1, 2, 3, 4, 5, 6]
*)
(* @assert:cpu:result
[6, 15]
*)
(* @assert:cpu:shape
[2]
*)
sum(a, axis=1)

(* @case columns *)
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: sum-i32-cpu
    arguments:
      - name: a
        dtype: i32
        shape: [2, 3]
        values: [1, 2, 3, 4, 5, 6]
*)
(* @assert:cpu:result
[5, 7, 9]
*)
(* @assert:cpu:shape
[3]
*)
sum(a, axis=0)

(* @case rank1-axis *)
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: sum-i32-cpu
    arguments:
      - name: a
        dtype: i32
        shape: [3]
        values: [1, 2, 3]
*)
(* @assert:cpu:result
[6]
*)
(* @assert:cpu:shape
[]
*)
sum(a, axis=0)

(* @case wrapping *)
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: sum-i32-cpu
    arguments:
      - name: a
        dtype: i32
        shape: [2, 3]
        values: [2147483647, 1, -2147483648, -1, -1, 2]
*)
(* @assert:cpu:result
[0, 0]
*)
(* @assert:cpu:shape
[2]
*)
sum(a, axis=1)

(* @case empty-0 *)
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: sum-i32-cpu
    arguments:
      - name: a
        dtype: i32
        shape: [0]
        values: []
*)
(* @assert:cpu:result
[0]
*)
(* @assert:cpu:shape
[]
*)
sum(a)

(* @case empty-1 *)
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: sum-i32-cpu
    arguments:
      - name: a
        dtype: i32
        shape: [2, 0]
        values: []
*)
(* @assert:cpu:result
[0, 0]
*)
(* @assert:cpu:shape
[2]
*)
sum(a, axis=1)

(* @case empty-2 *)
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: sum-i32-cpu
    arguments:
      - name: a
        dtype: i32
        shape: [0, 3]
        values: []
*)
(* @assert:cpu:result
[0, 0, 0]
*)
(* @assert:cpu:shape
[3]
*)
sum(a, axis=0)

(* @case empty-3 *)
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: sum-i32-cpu
    arguments:
      - name: a
        dtype: i32
        shape: [0, 3]
        values: []
*)
(* @assert:cpu:result
[]
*)
(* @assert:cpu:shape
[0]
*)
sum(a, axis=1)

(* @case empty-4 *)
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: sum-i32-cpu
    arguments:
      - name: a
        dtype: i32
        shape: [2, 0]
        values: []
*)
(* @assert:cpu:result
[]
*)
(* @assert:cpu:shape
[0]
*)
sum(a, axis=0)

(* @case empty-5 *)
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: sum-i32-cpu
    arguments:
      - name: a
        dtype: i32
        shape: [0, 0]
        values: []
*)
(* @assert:cpu:result
[0]
*)
(* @assert:cpu:shape
[]
*)
sum(a)

(* @case fractions *)
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: sum-f32-cpu
    profile: f32-sum-tree256-ftz
    arguments:
      - name: a
        dtype: f32
        shape: [2, 3]
        values: [0x3f000000, 0x3e800000, 0xbf000000, 0x3f800000, 0x40000000, 0x40400000]
*)
(* @assert:cpu:result
[0x3e800000, 0x40c00000]
*)
(* @assert:cpu:shape
[2]
*)
sum(a, axis=1)

(* @case cancellation *)
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: sum-f32-cpu
    profile: f32-sum-tree256-ftz
    arguments:
      - name: a
        dtype: f32
        shape: [4]
        values: [0x4b800000, 0x3f800000, 0x3f800000, 0xcb800000]
*)
(* @assert:cpu:result
[0x3f800000]
*)
(* @assert:cpu:shape
[]
*)
sum(a)

(* @case exceptional *)
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: sum-f32-cpu
    profile: f32-sum-tree256-ftz
    arguments:
      - name: a
        dtype: f32
        shape: [3, 2]
        values: [0x7f800000, 0xff800000, 0x7fc12345, 0x3f800000, 0x80000000, 0x80000000]
*)
(* @assert:cpu:result
[0x7fc00000, 0x7fc00000, 0x00000000]
*)
(* @assert:cpu:shape
[3]
*)
sum(a, axis=1)

(* @case ftz *)
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: sum-f32-cpu
    profile: f32-sum-tree256-ftz
    arguments:
      - name: a
        dtype: f32
        shape: [2, 2]
        values: [0x00000001, 0x80000001, 0x00800000, 0x80800001]
*)
(* @assert:cpu:result
[0x00000000, 0x00000000]
*)
(* @assert:cpu:shape
[2]
*)
sum(a, axis=1)

(* @case overflow *)
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: sum-f32-cpu
    profile: f32-sum-tree256-ftz
    arguments:
      - name: a
        dtype: f32
        shape: [2]
        values: [0x7f7fffff, 0x7f7fffff]
*)
(* @assert:cpu:result
[0x7f800000]
*)
(* @assert:cpu:shape
[]
*)
sum(a)

