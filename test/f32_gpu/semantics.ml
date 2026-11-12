(* @case row-broadcast *)
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: shaped-f32-cpu
    profile: f32-ftz
    arguments:
      - name: a
        dtype: f32
        values: [1, 2, 3, 4, 5, 6]
        shape: [2, 3]
      - name: b
        dtype: f32
        values: [0.25, -0.5, 1.5]
        shape: [3]
  - name: metal
    pipeline: shaped-f32-metal
    threads-per-group: 4
    profile: f32-ftz
    arguments:
      - name: a
        dtype: f32
        values: [1, 2, 3, 4, 5, 6]
        shape: [2, 3]
      - name: b
        dtype: f32
        values: [0.25, -0.5, 1.5]
        shape: [3]
*)
(* @assert:metal:result
[0.75, 0.5, 3, 2.25, 2, 4.5]
*)
(* @assert:metal:shape
[2, 3]
*)
(* @assert:metal:compare
cpu
*)
(* @assert:cpu:result
[0.75, 0.5, 3, 2.25, 2, 4.5]
*)
(* @assert:cpu:shape
[2, 3]
*)
(* @assert:cpu:ir-check
CHECK: llvm.fmul
CHECK: llvm.fadd
CHECK-NOT: fastmath<
*)
a * 0.5 + b

(* @case three-input *)
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: shaped-f32-cpu
    profile: f32-ftz
    arguments:
      - name: a
        dtype: f32
        values: [1.5, -2.5]
        shape: [2, 1]
      - name: b
        dtype: f32
        values: [0.5, 2, -1]
        shape: [1, 3]
      - name: c
        dtype: f32
        values: [0.25]
        shape: [1]
  - name: metal
    pipeline: shaped-f32-metal
    threads-per-group: 4
    profile: f32-ftz
    arguments:
      - name: a
        dtype: f32
        values: [1.5, -2.5]
        shape: [2, 1]
      - name: b
        dtype: f32
        values: [0.5, 2, -1]
        shape: [1, 3]
      - name: c
        dtype: f32
        values: [0.25]
        shape: [1]
*)
(* @assert:metal:result
[1, 3.25, -1.25, -1, -4.75, 2.75]
*)
(* @assert:metal:shape
[2, 3]
*)
(* @assert:metal:compare
cpu
*)
(* @assert:cpu:result
[1, 3.25, -1.25, -1, -4.75, 2.75]
*)
(* @assert:cpu:shape
[2, 3]
*)
a * b + c

(* @case constant *)
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: shaped-f32-cpu
    profile: f32-ftz
    arguments:
      - name: a
        dtype: f32
        values: [1, 2]
        shape: [2]
  - name: metal
    pipeline: shaped-f32-metal
    threads-per-group: 4
    profile: f32-ftz
    arguments:
      - name: a
        dtype: f32
        values: [1, 2]
        shape: [2]
*)
(* @assert:metal:result
[0x80000000, 0x80000000]
*)
(* @assert:metal:shape
[2]
*)
(* @assert:metal:compare
cpu
*)
(* @assert:cpu:result
[0x80000000, 0x80000000]
*)
(* @assert:cpu:shape
[2]
*)
-0.0

(* @case literal-rounding *)
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: shaped-f32-cpu
    profile: f32-ftz
    arguments:
      - name: a
        dtype: f32
        values: [0]
        shape: [1]
  - name: metal
    pipeline: shaped-f32-metal
    threads-per-group: 4
    profile: f32-ftz
    arguments:
      - name: a
        dtype: f32
        values: [0]
        shape: [1]
*)
(* @assert:metal:result
[0x3dcccccd]
*)
(* @assert:metal:shape
[1]
*)
(* @assert:metal:compare
cpu
*)
(* @assert:cpu:result
[0x3dcccccd]
*)
(* @assert:cpu:shape
[1]
*)
0.1

(* @case scientific *)
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: shaped-f32-cpu
    profile: f32-ftz
    arguments:
      - name: a
        dtype: f32
        values: [0.25, 0.5]
        shape: [2]
  - name: metal
    pipeline: shaped-f32-metal
    threads-per-group: 4
    profile: f32-ftz
    arguments:
      - name: a
        dtype: f32
        values: [0.25, 0.5]
        shape: [2]
*)
(* @assert:metal:result
[0.125, 0.375]
*)
(* @assert:metal:shape
[2]
*)
(* @assert:metal:compare
cpu
*)
(* @assert:cpu:result
[0.125, 0.375]
*)
(* @assert:cpu:shape
[2]
*)
a + -1.25e-1

(* @case left-associative *)
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: shaped-f32-cpu
    profile: f32-ftz
    arguments:
      - name: a
        dtype: f32
        values: [1]
        shape: [1]
      - name: b
        dtype: f32
        values: [0.25]
        shape: [1]
  - name: metal
    pipeline: shaped-f32-metal
    threads-per-group: 4
    profile: f32-ftz
    arguments:
      - name: a
        dtype: f32
        values: [1]
        shape: [1]
      - name: b
        dtype: f32
        values: [0.25]
        shape: [1]
*)
(* @assert:metal:result
[0.25]
*)
(* @assert:metal:shape
[1]
*)
(* @assert:metal:compare
cpu
*)
(* @assert:cpu:result
[0.25]
*)
(* @assert:cpu:shape
[1]
*)
a - b - 0.5

(* @case noncontracted *)
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: shaped-f32-cpu
    profile: f32-ftz
    arguments:
      - name: a
        dtype: f32
        values: [0x3f800001]
        shape: [1]
      - name: b
        dtype: f32
        values: [0x3f7ffffe]
        shape: [1]
      - name: c
        dtype: f32
        values: [-1]
        shape: [1]
  - name: metal
    pipeline: shaped-f32-metal
    threads-per-group: 4
    profile: f32-ftz
    arguments:
      - name: a
        dtype: f32
        values: [0x3f800001]
        shape: [1]
      - name: b
        dtype: f32
        values: [0x3f7ffffe]
        shape: [1]
      - name: c
        dtype: f32
        values: [-1]
        shape: [1]
*)
(* @assert:metal:result
[0x00000000]
*)
(* @assert:metal:shape
[1]
*)
(* @assert:metal:compare
cpu
*)
(* @assert:cpu:result
[0x00000000]
*)
(* @assert:cpu:shape
[1]
*)
a * b + c

(* @case round-even *)
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: shaped-f32-cpu
    profile: f32-ftz
    arguments:
      - name: a
        dtype: f32
        values: [0x3f800000, 0x3f800001]
        shape: [2]
      - name: b
        dtype: f32
        values: [0x33800000]
        shape: [1]
  - name: metal
    pipeline: shaped-f32-metal
    threads-per-group: 4
    profile: f32-ftz
    arguments:
      - name: a
        dtype: f32
        values: [0x3f800000, 0x3f800001]
        shape: [2]
      - name: b
        dtype: f32
        values: [0x33800000]
        shape: [1]
*)
(* @assert:metal:result
[0x3f800000, 0x3f800002]
*)
(* @assert:metal:shape
[2]
*)
(* @assert:metal:compare
cpu
*)
(* @assert:cpu:result
[0x3f800000, 0x3f800002]
*)
(* @assert:cpu:shape
[2]
*)
a + b

(* @case identity-ftz *)
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: shaped-f32-cpu
    profile: f32-ftz
    arguments:
      - name: a
        dtype: f32
        values: [0x00000001, 0x80000001, 0x00800000, 0x80800000, 0x7fc12345, inf, -inf, 0x80000000]
        shape: [8]
  - name: metal
    pipeline: shaped-f32-metal
    threads-per-group: 4
    profile: f32-ftz
    arguments:
      - name: a
        dtype: f32
        values: [0x00000001, 0x80000001, 0x00800000, 0x80800000, 0x7fc12345, inf, -inf, 0x80000000]
        shape: [8]
*)
(* @assert:metal:result
[0x00000000, 0x80000000, 0x00800000, 0x80800000, nan, inf, -inf, 0x80000000]
*)
(* @assert:metal:shape
[8]
*)
(* @assert:metal:compare
cpu
*)
(* @assert:cpu:result
[0x00000000, 0x80000000, 0x00800000, 0x80800000, nan, inf, -inf, 0x80000000]
*)
(* @assert:cpu:shape
[8]
*)
a

(* @case underflow *)
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: shaped-f32-cpu
    profile: f32-ftz
    arguments:
      - name: a
        dtype: f32
        values: [0x00800000, 0x80800000]
        shape: [2]
  - name: metal
    pipeline: shaped-f32-metal
    threads-per-group: 4
    profile: f32-ftz
    arguments:
      - name: a
        dtype: f32
        values: [0x00800000, 0x80800000]
        shape: [2]
*)
(* @assert:metal:result
[0x00000000, 0x80000000]
*)
(* @assert:metal:shape
[2]
*)
(* @assert:metal:compare
cpu
*)
(* @assert:cpu:result
[0x00000000, 0x80000000]
*)
(* @assert:cpu:shape
[2]
*)
a * 0.5

(* @case intermediate-ftz *)
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: shaped-f32-cpu
    profile: f32-ftz
    arguments:
      - name: a
        dtype: f32
        values: [0x00800000, 0x80800000]
        shape: [2]
  - name: metal
    pipeline: shaped-f32-metal
    threads-per-group: 4
    profile: f32-ftz
    arguments:
      - name: a
        dtype: f32
        values: [0x00800000, 0x80800000]
        shape: [2]
*)
(* @assert:metal:result
[0x00000000, 0x80000000]
*)
(* @assert:metal:shape
[2]
*)
(* @assert:metal:compare
cpu
*)
(* @assert:cpu:result
[0x00000000, 0x80000000]
*)
(* @assert:cpu:shape
[2]
*)
a * 0.5 * 16777216

(* @case literal-ftz *)
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: shaped-f32-cpu
    profile: f32-ftz
    arguments:
      - name: a
        dtype: f32
        values: [1]
        shape: [1]
  - name: metal
    pipeline: shaped-f32-metal
    threads-per-group: 4
    profile: f32-ftz
    arguments:
      - name: a
        dtype: f32
        values: [1]
        shape: [1]
*)
(* @assert:metal:result
[0x80000000]
*)
(* @assert:metal:shape
[1]
*)
(* @assert:metal:compare
cpu
*)
(* @assert:cpu:result
[0x80000000]
*)
(* @assert:cpu:shape
[1]
*)
a * -1e-40

(* @case exceptional *)
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: shaped-f32-cpu
    profile: f32-ftz
    arguments:
      - name: a
        dtype: f32
        values: [inf, -inf, nan, 0x7f7fffff, 0x80000000]
        shape: [5]
      - name: b
        dtype: f32
        values: [0, 2, 1, 2, 1]
        shape: [5]
  - name: metal
    pipeline: shaped-f32-metal
    threads-per-group: 4
    profile: f32-ftz
    arguments:
      - name: a
        dtype: f32
        values: [inf, -inf, nan, 0x7f7fffff, 0x80000000]
        shape: [5]
      - name: b
        dtype: f32
        values: [0, 2, 1, 2, 1]
        shape: [5]
*)
(* @assert:metal:result
[nan, -inf, nan, inf, 0x80000000]
*)
(* @assert:metal:shape
[5]
*)
(* @assert:metal:compare
cpu
*)
(* @assert:cpu:result
[nan, -inf, nan, inf, 0x80000000]
*)
(* @assert:cpu:shape
[5]
*)
a * b

(* @case cancellation *)
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: shaped-f32-cpu
    profile: f32-ftz
    arguments:
      - name: a
        dtype: f32
        values: [0x00800001, 0x80800001, 1]
        shape: [3]
      - name: b
        dtype: f32
        values: [0x00800000, 0x80800000, 1]
        shape: [3]
  - name: metal
    pipeline: shaped-f32-metal
    threads-per-group: 4
    profile: f32-ftz
    arguments:
      - name: a
        dtype: f32
        values: [0x00800001, 0x80800001, 1]
        shape: [3]
      - name: b
        dtype: f32
        values: [0x00800000, 0x80800000, 1]
        shape: [3]
*)
(* @assert:metal:result
[0x00000000, 0x80000000, 0x00000000]
*)
(* @assert:metal:shape
[3]
*)
(* @assert:metal:compare
cpu
*)
(* @assert:cpu:result
[0x00000000, 0x80000000, 0x00000000]
*)
(* @assert:cpu:shape
[3]
*)
a - b

(* @case empty-row *)
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: shaped-f32-cpu
    profile: f32-ftz
    arguments:
      - name: a
        dtype: f32
        values: []
        shape: [0, 3]
      - name: b
        dtype: f32
        values: [1, 2, 3]
        shape: [3]
  - name: metal
    pipeline: shaped-f32-metal
    threads-per-group: 4
    profile: f32-ftz
    arguments:
      - name: a
        dtype: f32
        values: []
        shape: [0, 3]
      - name: b
        dtype: f32
        values: [1, 2, 3]
        shape: [3]
*)
(* @assert:metal:result
[]
*)
(* @assert:metal:shape
[0, 3]
*)
(* @assert:metal:compare
cpu
*)
(* @assert:cpu:result
[]
*)
(* @assert:cpu:shape
[0, 3]
*)
a * 0.5 + b

(* @case tininess-before-rounding *)
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: shaped-f32-cpu
    profile: f32-ftz
    arguments:
      - name: a
        dtype: f32
        values: [0x00800000, 0x80800000, 0x00800001, 0x00800000]
        shape: [4]
      - name: b
        dtype: f32
        values: [0x3f7fffff, 0x3f7fffff, 0x3f7ffffe, 1.0]
        shape: [4]
  - name: metal
    pipeline: shaped-f32-metal
    threads-per-group: 4
    profile: f32-ftz
    arguments:
      - name: a
        dtype: f32
        values: [0x00800000, 0x80800000, 0x00800001, 0x00800000]
        shape: [4]
      - name: b
        dtype: f32
        values: [0x3f7fffff, 0x3f7fffff, 0x3f7ffffe, 1.0]
        shape: [4]
*)
(* @assert:metal:result
[0x00000000, 0x80000000, 0x00000000, 0x00800000]
*)
(* @assert:metal:shape
[4]
*)
(* @assert:metal:compare
cpu
*)
(* @assert:cpu:result
[0x00000000, 0x80000000, 0x00000000, 0x00800000]
*)
(* @assert:cpu:shape
[4]
*)
a * b
