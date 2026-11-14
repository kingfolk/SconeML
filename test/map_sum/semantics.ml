(* @case sample-i32-None *)
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
      - name: b
        dtype: i32
        shape: [3]
        values: [10, 20, 30]
  - name: metal
    pipeline: sum-i32-metal
    arguments:
      - name: a
        dtype: i32
        shape: [2, 3]
        values: [1, 2, 3, 4, 5, 6]
      - name: b
        dtype: i32
        shape: [3]
        values: [10, 20, 30]
*)
(* @assert:cpu:result
[460]
*)
(* @assert:cpu:shape
[]
*)
(* @assert:metal:result
[460]
*)
(* @assert:metal:shape
[]
*)
(* @assert:metal:compare
cpu
*)
sum(a * b)

(* @case sample-i32-0 *)
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
      - name: b
        dtype: i32
        shape: [3]
        values: [10, 20, 30]
  - name: metal
    pipeline: sum-i32-metal
    arguments:
      - name: a
        dtype: i32
        shape: [2, 3]
        values: [1, 2, 3, 4, 5, 6]
      - name: b
        dtype: i32
        shape: [3]
        values: [10, 20, 30]
*)
(* @assert:cpu:result
[50, 140, 270]
*)
(* @assert:cpu:shape
[3]
*)
(* @assert:metal:result
[50, 140, 270]
*)
(* @assert:metal:shape
[3]
*)
(* @assert:metal:compare
cpu
*)
sum(a * b, axis=0)

(* @case sample-i32-1 *)
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
      - name: b
        dtype: i32
        shape: [3]
        values: [10, 20, 30]
  - name: metal
    pipeline: sum-i32-metal
    arguments:
      - name: a
        dtype: i32
        shape: [2, 3]
        values: [1, 2, 3, 4, 5, 6]
      - name: b
        dtype: i32
        shape: [3]
        values: [10, 20, 30]
*)
(* @assert:cpu:result
[140, 320]
*)
(* @assert:cpu:shape
[2]
*)
(* @assert:metal:result
[140, 320]
*)
(* @assert:metal:shape
[2]
*)
(* @assert:metal:compare
cpu
*)
sum(a * b, axis=1)

(* @case overlap-i32 *)
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
      - name: b
        dtype: i32
        shape: [3]
        values: [2, 3, 4]
  - name: metal
    pipeline: sum-i32-metal
    arguments:
      - name: a
        dtype: i32
        shape: [2, 3]
        values: [1, 2, 3, 4, 5, 6]
      - name: b
        dtype: i32
        shape: [3]
        values: [2, 3, 4]
*)
(* @assert:cpu:result
[20, 47]
*)
(* @assert:cpu:shape
[2]
*)
(* @assert:metal:result
[20, 47]
*)
(* @assert:metal:shape
[2]
*)
(* @assert:metal:compare
cpu
*)
sum(a * b, axis=1)

(* @case three-i32 *)
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
      - name: b
        dtype: i32
        shape: [3]
        values: [10, 20, 30]
      - name: c
        dtype: i32
        shape: [2, 1]
        values: [1, -1]
  - name: metal
    pipeline: sum-i32-metal
    arguments:
      - name: a
        dtype: i32
        shape: [2, 3]
        values: [1, 2, 3, 4, 5, 6]
      - name: b
        dtype: i32
        shape: [3]
        values: [10, 20, 30]
      - name: c
        dtype: i32
        shape: [2, 1]
        values: [1, -1]
*)
(* @assert:cpu:result
[143, 317]
*)
(* @assert:cpu:shape
[2]
*)
(* @assert:metal:result
[143, 317]
*)
(* @assert:metal:shape
[2]
*)
(* @assert:metal:compare
cpu
*)
sum(a * b + c, axis=1)

(* @case empty-i32 *)
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
      - name: b
        dtype: i32
        shape: [1]
        values: [1]
  - name: metal
    pipeline: sum-i32-metal
    arguments:
      - name: a
        dtype: i32
        shape: [2, 0]
        values: []
      - name: b
        dtype: i32
        shape: [1]
        values: [1]
*)
(* @assert:cpu:result
[0, 0]
*)
(* @assert:cpu:shape
[2]
*)
(* @assert:metal:result
[0, 0]
*)
(* @assert:metal:shape
[2]
*)
(* @assert:metal:compare
cpu
*)
sum(a * b, axis=1)

(* @case zero-output-i32 *)
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
      - name: b
        dtype: i32
        shape: [3]
        values: [1, 2, 3]
  - name: metal
    pipeline: sum-i32-metal
    arguments:
      - name: a
        dtype: i32
        shape: [0, 3]
        values: []
      - name: b
        dtype: i32
        shape: [3]
        values: [1, 2, 3]
*)
(* @assert:cpu:result
[]
*)
(* @assert:cpu:shape
[0]
*)
(* @assert:metal:result
[]
*)
(* @assert:metal:shape
[0]
*)
(* @assert:metal:compare
cpu
*)
sum(a * b, axis=1)

(* @case unused-i32 *)
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
      - name: b
        dtype: i32
        shape: [2, 1]
        values: [4, 5]
  - name: metal
    pipeline: sum-i32-metal
    arguments:
      - name: a
        dtype: i32
        shape: [3]
        values: [1, 2, 3]
      - name: b
        dtype: i32
        shape: [2, 1]
        values: [4, 5]
*)
(* @assert:cpu:result
[4, 8, 12]
*)
(* @assert:cpu:shape
[3]
*)
(* @assert:metal:result
[4, 8, 12]
*)
(* @assert:metal:shape
[3]
*)
(* @assert:metal:compare
cpu
*)
sum(a + a, axis=0)

(* @case sample-f32-None *)
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
        values: [0x3f800000, 0x40000000, 0x40400000, 0x40800000, 0x40a00000, 0x40c00000]
      - name: b
        dtype: f32
        shape: [3]
        values: [0x41200000, 0x41a00000, 0x41f00000]
  - name: metal
    pipeline: sum-f32-metal
    profile: f32-sum-tree256-ftz
    arguments:
      - name: a
        dtype: f32
        shape: [2, 3]
        values: [0x3f800000, 0x40000000, 0x40400000, 0x40800000, 0x40a00000, 0x40c00000]
      - name: b
        dtype: f32
        shape: [3]
        values: [0x41200000, 0x41a00000, 0x41f00000]
*)
(* @assert:cpu:result
[0x43e60000]
*)
(* @assert:cpu:shape
[]
*)
(* @assert:metal:result
[0x43e60000]
*)
(* @assert:metal:shape
[]
*)
(* @assert:metal:compare
cpu
*)
sum(a * b)

(* @case sample-f32-0 *)
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
        values: [0x3f800000, 0x40000000, 0x40400000, 0x40800000, 0x40a00000, 0x40c00000]
      - name: b
        dtype: f32
        shape: [3]
        values: [0x41200000, 0x41a00000, 0x41f00000]
  - name: metal
    pipeline: sum-f32-metal
    profile: f32-sum-tree256-ftz
    arguments:
      - name: a
        dtype: f32
        shape: [2, 3]
        values: [0x3f800000, 0x40000000, 0x40400000, 0x40800000, 0x40a00000, 0x40c00000]
      - name: b
        dtype: f32
        shape: [3]
        values: [0x41200000, 0x41a00000, 0x41f00000]
*)
(* @assert:cpu:result
[0x42480000, 0x430c0000, 0x43870000]
*)
(* @assert:cpu:shape
[3]
*)
(* @assert:metal:result
[0x42480000, 0x430c0000, 0x43870000]
*)
(* @assert:metal:shape
[3]
*)
(* @assert:metal:compare
cpu
*)
sum(a * b, axis=0)

(* @case sample-f32-1 *)
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
        values: [0x3f800000, 0x40000000, 0x40400000, 0x40800000, 0x40a00000, 0x40c00000]
      - name: b
        dtype: f32
        shape: [3]
        values: [0x41200000, 0x41a00000, 0x41f00000]
  - name: metal
    pipeline: sum-f32-metal
    profile: f32-sum-tree256-ftz
    arguments:
      - name: a
        dtype: f32
        shape: [2, 3]
        values: [0x3f800000, 0x40000000, 0x40400000, 0x40800000, 0x40a00000, 0x40c00000]
      - name: b
        dtype: f32
        shape: [3]
        values: [0x41200000, 0x41a00000, 0x41f00000]
*)
(* @assert:cpu:result
[0x430c0000, 0x43a00000]
*)
(* @assert:cpu:shape
[2]
*)
(* @assert:metal:result
[0x430c0000, 0x43a00000]
*)
(* @assert:metal:shape
[2]
*)
(* @assert:metal:compare
cpu
*)
sum(a * b, axis=1)

(* @case overlap-f32 *)
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
        values: [0x3f800000, 0x40000000, 0x40400000, 0x40800000, 0x40a00000, 0x40c00000]
      - name: b
        dtype: f32
        shape: [3]
        values: [0x40000000, 0x40400000, 0x40800000]
  - name: metal
    pipeline: sum-f32-metal
    profile: f32-sum-tree256-ftz
    arguments:
      - name: a
        dtype: f32
        shape: [2, 3]
        values: [0x3f800000, 0x40000000, 0x40400000, 0x40800000, 0x40a00000, 0x40c00000]
      - name: b
        dtype: f32
        shape: [3]
        values: [0x40000000, 0x40400000, 0x40800000]
*)
(* @assert:cpu:result
[0x41a00000, 0x423c0000]
*)
(* @assert:cpu:shape
[2]
*)
(* @assert:metal:result
[0x41a00000, 0x423c0000]
*)
(* @assert:metal:shape
[2]
*)
(* @assert:metal:compare
cpu
*)
sum(a * b, axis=1)

(* @case three-f32 *)
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
        values: [0x3f800000, 0x40000000, 0x40400000, 0x40800000, 0x40a00000, 0x40c00000]
      - name: b
        dtype: f32
        shape: [3]
        values: [0x41200000, 0x41a00000, 0x41f00000]
      - name: c
        dtype: f32
        shape: [2, 1]
        values: [0x3f800000, 0xbf800000]
  - name: metal
    pipeline: sum-f32-metal
    profile: f32-sum-tree256-ftz
    arguments:
      - name: a
        dtype: f32
        shape: [2, 3]
        values: [0x3f800000, 0x40000000, 0x40400000, 0x40800000, 0x40a00000, 0x40c00000]
      - name: b
        dtype: f32
        shape: [3]
        values: [0x41200000, 0x41a00000, 0x41f00000]
      - name: c
        dtype: f32
        shape: [2, 1]
        values: [0x3f800000, 0xbf800000]
*)
(* @assert:cpu:result
[0x430f0000, 0x439e8000]
*)
(* @assert:cpu:shape
[2]
*)
(* @assert:metal:result
[0x430f0000, 0x439e8000]
*)
(* @assert:metal:shape
[2]
*)
(* @assert:metal:compare
cpu
*)
sum(a * b + c, axis=1)

(* @case empty-f32 *)
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
        shape: [2, 0]
        values: []
      - name: b
        dtype: f32
        shape: [1]
        values: [0x3f800000]
  - name: metal
    pipeline: sum-f32-metal
    profile: f32-sum-tree256-ftz
    arguments:
      - name: a
        dtype: f32
        shape: [2, 0]
        values: []
      - name: b
        dtype: f32
        shape: [1]
        values: [0x3f800000]
*)
(* @assert:cpu:result
[0x00000000, 0x00000000]
*)
(* @assert:cpu:shape
[2]
*)
(* @assert:metal:result
[0x00000000, 0x00000000]
*)
(* @assert:metal:shape
[2]
*)
(* @assert:metal:compare
cpu
*)
sum(a * b, axis=1)

(* @case zero-output-f32 *)
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
        shape: [0, 3]
        values: []
      - name: b
        dtype: f32
        shape: [3]
        values: [0x3f800000, 0x40000000, 0x40400000]
  - name: metal
    pipeline: sum-f32-metal
    profile: f32-sum-tree256-ftz
    arguments:
      - name: a
        dtype: f32
        shape: [0, 3]
        values: []
      - name: b
        dtype: f32
        shape: [3]
        values: [0x3f800000, 0x40000000, 0x40400000]
*)
(* @assert:cpu:result
[]
*)
(* @assert:cpu:shape
[0]
*)
(* @assert:metal:result
[]
*)
(* @assert:metal:shape
[0]
*)
(* @assert:metal:compare
cpu
*)
sum(a * b, axis=1)

(* @case unused-f32 *)
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
        shape: [3]
        values: [0x3f800000, 0x40000000, 0x40400000]
      - name: b
        dtype: f32
        shape: [2, 1]
        values: [0x40800000, 0x40a00000]
  - name: metal
    pipeline: sum-f32-metal
    profile: f32-sum-tree256-ftz
    arguments:
      - name: a
        dtype: f32
        shape: [3]
        values: [0x3f800000, 0x40000000, 0x40400000]
      - name: b
        dtype: f32
        shape: [2, 1]
        values: [0x40800000, 0x40a00000]
*)
(* @assert:cpu:result
[0x40800000, 0x41000000, 0x41400000]
*)
(* @assert:cpu:shape
[3]
*)
(* @assert:metal:result
[0x40800000, 0x41000000, 0x41400000]
*)
(* @assert:metal:shape
[3]
*)
(* @assert:metal:compare
cpu
*)
sum(a + a, axis=0)

