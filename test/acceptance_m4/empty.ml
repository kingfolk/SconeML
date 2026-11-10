(* @case empty-0 *)
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
        shape: [0]
        values: []
      - name: b
        dtype: f32
        shape: [1]
        values: [0x7da3f1a0]
  - name: metal
    pipeline: shaped-f32-metal
    profile: f32-ftz
    threads-per-group: 32
    arguments:
      - name: a
        dtype: f32
        shape: [0]
        values: []
      - name: b
        dtype: f32
        shape: [1]
        values: [0x7da3f1a0]
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
a * 0.5 + b

(* @case empty-1 *)
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
        shape: [0, 3]
        values: []
      - name: b
        dtype: f32
        shape: [3]
        values: [0x7dbd57ad, 0x9fe05528, 0x0da93667]
  - name: metal
    pipeline: shaped-f32-metal
    profile: f32-ftz
    threads-per-group: 32
    arguments:
      - name: a
        dtype: f32
        shape: [0, 3]
        values: []
      - name: b
        dtype: f32
        shape: [3]
        values: [0x7dbd57ad, 0x9fe05528, 0x0da93667]
*)
(* @assert:cpu:result
[]
*)
(* @assert:cpu:shape
[0, 3]
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
a * 0.5 + b

(* @case empty-2 *)
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
        shape: [2, 0]
        values: []
      - name: b
        dtype: f32
        shape: [1, 0]
        values: []
  - name: metal
    pipeline: shaped-f32-metal
    profile: f32-ftz
    threads-per-group: 32
    arguments:
      - name: a
        dtype: f32
        shape: [2, 0]
        values: []
      - name: b
        dtype: f32
        shape: [1, 0]
        values: []
*)
(* @assert:cpu:result
[]
*)
(* @assert:cpu:shape
[2, 0]
*)
(* @assert:metal:result
[]
*)
(* @assert:metal:shape
[2, 0]
*)
(* @assert:metal:compare
cpu
*)
a * 0.5 + b

(* @case empty-3 *)
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
        shape: [0, 0]
        values: []
      - name: b
        dtype: f32
        shape: [1]
        values: [0x7d24f35f]
  - name: metal
    pipeline: shaped-f32-metal
    profile: f32-ftz
    threads-per-group: 32
    arguments:
      - name: a
        dtype: f32
        shape: [0, 0]
        values: []
      - name: b
        dtype: f32
        shape: [1]
        values: [0x7d24f35f]
*)
(* @assert:cpu:result
[]
*)
(* @assert:cpu:shape
[0, 0]
*)
(* @assert:metal:result
[]
*)
(* @assert:metal:shape
[0, 0]
*)
(* @assert:metal:compare
cpu
*)
a * 0.5 + b

(* @case empty-4 *)
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
        shape: [0, 1]
        values: []
      - name: b
        dtype: f32
        shape: [1, 3]
        values: [0x7d3e596c, 0x2bc685db, 0xa13c017e]
  - name: metal
    pipeline: shaped-f32-metal
    profile: f32-ftz
    threads-per-group: 32
    arguments:
      - name: a
        dtype: f32
        shape: [0, 1]
        values: []
      - name: b
        dtype: f32
        shape: [1, 3]
        values: [0x7d3e596c, 0x2bc685db, 0xa13c017e]
*)
(* @assert:cpu:result
[]
*)
(* @assert:cpu:shape
[0, 3]
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
a * 0.5 + b

(* @case empty-5 *)
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
        shape: [2, 0]
        values: []
      - name: b
        dtype: f32
        shape: [1]
        values: [0x7d57bf79]
  - name: metal
    pipeline: shaped-f32-metal
    profile: f32-ftz
    threads-per-group: 32
    arguments:
      - name: a
        dtype: f32
        shape: [2, 0]
        values: []
      - name: b
        dtype: f32
        shape: [1]
        values: [0x7d57bf79]
*)
(* @assert:cpu:result
[]
*)
(* @assert:cpu:shape
[2, 0]
*)
(* @assert:metal:result
[]
*)
(* @assert:metal:shape
[2, 0]
*)
(* @assert:metal:compare
cpu
*)
a * 0.5 + b

