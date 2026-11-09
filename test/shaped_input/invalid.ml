(* @case rank0 *)
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: shaped-input-cpu
    arguments:
      - name: a
        values: [1]
        shape: []
*)
(* @assert:cpu:error
stage: frontend
message: rank must be 1 or 2
*)
a

(* @case rank3 *)
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: shaped-input-cpu
    arguments:
      - name: a
        values: [1]
        shape: [1, 1, 1]
*)
(* @assert:cpu:error
stage: frontend
message: rank must be 1 or 2
*)
a

(* @case negative *)
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: shaped-input-cpu
    arguments:
      - name: a
        values: []
        shape: [-1, 3]
*)
(* @assert:cpu:error
stage: frontend
message: extents must be nonnegative
*)
a

(* @case negative-empty *)
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: shaped-input-cpu
    arguments:
      - name: a
        values: []
        shape: [0, -3]
*)
(* @assert:cpu:error
stage: frontend
message: extents must be nonnegative
*)
a

(* @case missing-storage *)
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: shaped-input-cpu
    arguments:
      - name: a
        values: [1]
        shape: [2, 3]
*)
(* @assert:cpu:error
stage: frontend
message: storage count does not match
*)
a

(* @case extra-storage *)
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: shaped-input-cpu
    arguments:
      - name: a
        values: [1, 2]
        shape: [1]
*)
(* @assert:cpu:error
stage: frontend
message: storage count does not match
*)
a

(* @case empty-extra-storage *)
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: shaped-input-cpu
    arguments:
      - name: a
        values: [1]
        shape: [0, 3]
*)
(* @assert:cpu:error
stage: frontend
message: storage count does not match
*)
a

(* @case same-count-different-shape *)
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: shaped-input-cpu
    arguments:
      - name: a
        values: [1, 2, 3, 4]
        shape: [2, 2]
      - name: b
        values: [5, 6, 7, 8]
        shape: [1, 4]
*)
(* @assert:cpu:error
stage: frontend
message: incompatible shaped tensor broadcast dimensions
*)
a + b

(* @case rank1-trailing-axis-mismatch *)
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: shaped-input-cpu
    arguments:
      - name: a
        values: [1, 2, 3, 4, 5, 6]
        shape: [2, 3]
      - name: b
        values: [5, 6]
        shape: [2]
*)
(* @assert:cpu:error
stage: frontend
message: incompatible shaped tensor broadcast dimensions
*)
a + b

(* @case unused-invalid *)
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: shaped-input-cpu
    arguments:
      - name: a
        values: [1]
        shape: [1]
      - name: b
        values: [2]
        shape: [-1]
*)
(* @assert:cpu:error
stage: frontend
message: extents must be nonnegative
*)
a

(* @case constant-invalid *)
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: shaped-input-cpu
    arguments:
      - name: a
        values: [1, 2]
        shape: [2]
      - name: b
        values: []
        shape: [0]
*)
(* @assert:cpu:error
stage: frontend
message: incompatible shaped tensor broadcast dimensions
*)
7

(* @case empty-mismatch *)
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: shaped-input-cpu
    arguments:
      - name: a
        values: []
        shape: [0, 3]
      - name: b
        values: []
        shape: [0, 2]
*)
(* @assert:cpu:error
stage: frontend
message: incompatible shaped tensor broadcast dimensions
*)
a

(* @case product-overflow *)
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: shaped-input-cpu
    arguments:
      - name: a
        values: []
        shape: [9223372036854775807, 2]
*)
(* @assert:cpu:error
stage: frontend
message: exceeds host index/byte/capacity range
*)
a

(* @case stride-overflow-empty *)
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: shaped-input-cpu
    arguments:
      - name: a
        values: []
        shape: [0, 9223372036854775807]
*)
(* @assert:cpu:error
stage: frontend
message: exceeds host index/byte/capacity range
*)
a

(* @case duplicate-name *)
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: shaped-input-cpu
    arguments:
      - name: a
        values: [1]
        shape: [1]
      - name: a
        values: [2]
        shape: [1]
*)
(* @assert:cpu:error
stage: frontend
message: duplicate tensor argument name
*)
a

(* @case invalid-name *)
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: shaped-input-cpu
    arguments:
      - name: let
        values: [1]
        shape: [1]
*)
(* @assert:cpu:error
stage: frontend
message: invalid tensor argument name
*)
a

(* @case unknown-name *)
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: shaped-input-cpu
    arguments:
      - name: a
        values: [1]
        shape: [1]
*)
(* @assert:cpu:error
stage: frontend
message: unknown tensor expression variable
*)
b

(* @case empty-unknown-name *)
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: shaped-input-cpu
    arguments:
      - name: a
        values: []
        shape: [0, 2]
*)
(* @assert:cpu:error
stage: frontend
message: unknown tensor expression variable
*)
b

(* @case empty-bad-syntax *)
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: shaped-input-cpu
    arguments:
      - name: a
        values: []
        shape: [0]
*)
(* @assert:cpu:error
stage: frontend
message: empty or missing an operand
*)
a +

(* @case no-arguments *)
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: shaped-input-cpu
    arguments: []
*)
(* @assert:cpu:error
stage: frontend
message: between one and three tensor arguments
*)
7

(* @case too-many *)
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: shaped-input-cpu
    arguments:
      - name: a
        values: [1]
        shape: [1]
      - name: b
        values: [1]
        shape: [1]
      - name: c
        values: [1]
        shape: [1]
      - name: d
        values: [1]
        shape: [1]
*)
(* @assert:cpu:error
stage: frontend
message: between one and three tensor arguments
*)
a
