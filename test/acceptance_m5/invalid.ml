(* @case bad-storage *)
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
        values: [1]
  - name: metal
    pipeline: sum-i32-metal
    arguments:
      - name: a
        dtype: i32
        shape: [0, 3]
        values: [1]
*)
(* @assert:cpu:error
stage: frontend
message: "storage count does not match shape product"
*)
(* @assert:metal:error
stage: frontend
message: "storage count does not match shape product"
*)
sum(a, axis=1)

(* @case bad-rank *)
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: sum-i32-cpu
    arguments:
      - name: a
        dtype: i32
        shape: [0, 1, 1]
        values: []
  - name: metal
    pipeline: sum-i32-metal
    arguments:
      - name: a
        dtype: i32
        shape: [0, 1, 1]
        values: []
*)
(* @assert:cpu:error
stage: frontend
message: "rank must be 1 or 2"
*)
(* @assert:metal:error
stage: frontend
message: "rank must be 1 or 2"
*)
sum(a)

(* @case duplicate-name *)
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
      - name: a
        dtype: i32
        shape: [0, 3]
        values: []
  - name: metal
    pipeline: sum-i32-metal
    arguments:
      - name: a
        dtype: i32
        shape: [0, 3]
        values: []
      - name: a
        dtype: i32
        shape: [0, 3]
        values: []
*)
(* @assert:cpu:error
stage: frontend
message: "duplicate tensor argument name"
*)
(* @assert:metal:error
stage: frontend
message: "duplicate tensor argument name"
*)
sum(a)

(* @case no-inputs *)
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: sum-i32-cpu
    arguments: []
  - name: metal
    pipeline: sum-i32-metal
    arguments: []
*)
(* @assert:cpu:error
stage: frontend
message: "between one and three"
*)
(* @assert:metal:error
stage: frontend
message: "between one and three"
*)
sum(1)

(* @case bad-broadcast *)
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
        shape: [2]
        values: [1, 2]
  - name: metal
    pipeline: sum-i32-metal
    arguments:
      - name: a
        dtype: i32
        shape: [0, 3]
        values: []
      - name: b
        dtype: i32
        shape: [2]
        values: [1, 2]
*)
(* @assert:cpu:error
stage: frontend
message: "incompatible shaped tensor broadcast"
*)
(* @assert:metal:error
stage: frontend
message: "incompatible shaped tensor broadcast"
*)
sum(a + b, axis=1)

(* @case reserved-name *)
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: sum-i32-cpu
    arguments:
      - name: let
        dtype: i32
        shape: [0]
        values: []
  - name: metal
    pipeline: sum-i32-metal
    arguments:
      - name: let
        dtype: i32
        shape: [0]
        values: []
*)
(* @assert:cpu:error
stage: frontend
message: "invalid tensor argument name"
*)
(* @assert:metal:error
stage: frontend
message: "invalid tensor argument name"
*)
sum(let)

(* @case missing-map-operand *)
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
  - name: metal
    pipeline: sum-i32-metal
    arguments:
      - name: a
        dtype: i32
        shape: [0, 3]
        values: []
*)
(* @assert:cpu:error
stage: frontend
message: "missing an operand"
*)
(* @assert:metal:error
stage: frontend
message: "missing an operand"
*)
sum(a +, axis=1)

(* @case fractional-axis *)
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
  - name: metal
    pipeline: sum-i32-metal
    arguments:
      - name: a
        dtype: i32
        shape: [0, 3]
        values: []
*)
(* @assert:cpu:error
stage: frontend
message: "nonnegative integer"
*)
(* @assert:metal:error
stage: frontend
message: "nonnegative integer"
*)
sum(a, axis=1.0)

(* @case literal-overflow-empty *)
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
  - name: metal
    pipeline: sum-f32-metal
    profile: f32-sum-tree256-ftz
    arguments:
      - name: a
        dtype: f32
        shape: [0, 3]
        values: []
*)
(* @assert:cpu:error
stage: frontend
message: "outside finite binary32 range"
*)
(* @assert:metal:error
stage: frontend
message: "outside finite binary32 range"
*)
sum(a * 1e100, axis=1)

(* @case nested-reduction *)
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
  - name: metal
    pipeline: sum-i32-metal
    arguments:
      - name: a
        dtype: i32
        shape: [0, 3]
        values: []
*)
(* @assert:cpu:error
stage: frontend
message: "expected +, - or *"
*)
(* @assert:metal:error
stage: frontend
message: "expected +, - or *"
*)
sum(sum(a))

