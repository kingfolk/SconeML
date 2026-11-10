(* @case overflow-literal *)
(* @test
version: 1
input: source
runs:
  - name: metal
    pipeline: shaped-f32-metal
    profile: f32-ftz
    arguments:
      - name: a
        dtype: f32
        values: []
        shape: [0]
*)
(* @assert:metal:error
stage: frontend
message: outside finite binary32 range
*)
a + 1e100

(* @case invalid-exponent *)
(* @test
version: 1
input: source
runs:
  - name: metal
    pipeline: shaped-f32-metal
    profile: f32-ftz
    arguments:
      - name: a
        dtype: f32
        values: []
        shape: [0]
*)
(* @assert:metal:error
stage: frontend
message: invalid f32 literal exponent
*)
a * 1e-

(* @case invalid-decimal *)
(* @test
version: 1
input: source
runs:
  - name: metal
    pipeline: shaped-f32-metal
    profile: f32-ftz
    arguments:
      - name: a
        dtype: f32
        values: []
        shape: [0]
*)
(* @assert:metal:error
stage: frontend
message: invalid f32 literal
*)
a * .

(* @case unsupported-division *)
(* @test
version: 1
input: source
runs:
  - name: metal
    pipeline: shaped-f32-metal
    profile: f32-ftz
    arguments:
      - name: a
        dtype: f32
        values: []
        shape: [0]
*)
(* @assert:metal:error
stage: frontend
message: expected +, - or *
*)
a / 2.0

(* @case unknown-name *)
(* @test
version: 1
input: source
runs:
  - name: metal
    pipeline: shaped-f32-metal
    profile: f32-ftz
    arguments:
      - name: a
        dtype: f32
        values: []
        shape: [0]
*)
(* @assert:metal:error
stage: frontend
message: unknown tensor expression variable
*)
a + b

(* @case unsupported-cast *)
(* @test
version: 1
input: source
runs:
  - name: metal
    pipeline: shaped-f32-metal
    profile: f32-ftz
    arguments:
      - name: a
        dtype: f32
        values: []
        shape: [0]
*)
(* @assert:metal:error
stage: frontend
message: expected +, - or *
*)
f32(a)

(* @case invalid-shape *)
(* @test
version: 1
input: source
runs:
  - name: metal
    pipeline: shaped-f32-metal
    profile: f32-ftz
    arguments:
      - name: a
        dtype: f32
        values: [1]
        shape: [2]
*)
(* @assert:metal:error
stage: frontend
message: storage count does not match shape product
*)
a

(* @case incompatible *)
(* @test
version: 1
input: source
runs:
  - name: metal
    pipeline: shaped-f32-metal
    profile: f32-ftz
    arguments:
      - name: a
        dtype: f32
        values: [1, 2]
        shape: [2]
      - name: b
        dtype: f32
        values: [1, 2, 3]
        shape: [3]
*)
(* @assert:metal:error
stage: frontend
message: incompatible shaped tensor broadcast dimensions
*)
a + b

(* @case missing-operand *)
(* @test
version: 1
input: source
runs:
  - name: metal
    pipeline: shaped-f32-metal
    profile: f32-ftz
    arguments:
      - name: a
        dtype: f32
        values: []
        shape: [0]
*)
(* @assert:metal:error
stage: frontend
message: missing an operand
*)
a +

(* @case no-inputs *)
(* @test
version: 1
input: source
runs:
  - name: metal
    pipeline: shaped-f32-metal
    profile: f32-ftz
    arguments: []
*)
(* @assert:metal:error
stage: frontend
message: between one and three
*)
1.0

(* @case nan-literal *)
(* @test
version: 1
input: source
runs:
  - name: metal
    pipeline: shaped-f32-metal
    profile: f32-ftz
    arguments:
      - name: a
        dtype: f32
        values: []
        shape: [0]
*)
(* @assert:metal:error
stage: frontend
message: finite decimal literal
*)
nan

(* @case duplicate *)
(* @test
version: 1
input: source
runs:
  - name: metal
    pipeline: shaped-f32-metal
    profile: f32-ftz
    arguments:
      - name: a
        dtype: f32
        values: []
        shape: [0]
      - name: a
        dtype: f32
        values: []
        shape: [0]
*)
(* @assert:metal:error
stage: frontend
message: duplicate tensor argument name
*)
a

