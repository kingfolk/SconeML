(* @case zero *)
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: tensor-input-cpu
    arguments: []
  - name: metal
    pipeline: tensor-input-metal
    threads-per-group: 4
    arguments: []
*)
(* @assert:cpu:error
stage: frontend
message: "requires between one and three tensor arguments"
*)
(* @assert:metal:error
stage: frontend
message: "requires between one and three tensor arguments"
*)
a

(* @case four *)
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: tensor-input-cpu
    arguments:
      - name: "a"
        values: [1]
      - name: "b"
        values: [2]
      - name: "c"
        values: [3]
      - name: "d"
        values: [4]
  - name: metal
    pipeline: tensor-input-metal
    threads-per-group: 4
    arguments:
      - name: "a"
        values: [1]
      - name: "b"
        values: [2]
      - name: "c"
        values: [3]
      - name: "d"
        values: [4]
*)
(* @assert:cpu:error
stage: frontend
message: "requires between one and three tensor arguments"
*)
(* @assert:metal:error
stage: frontend
message: "requires between one and three tensor arguments"
*)
a

(* @case duplicate *)
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: tensor-input-cpu
    arguments:
      - name: "a"
        values: [1]
      - name: "a"
        values: [2]
  - name: metal
    pipeline: tensor-input-metal
    threads-per-group: 4
    arguments:
      - name: "a"
        values: [1]
      - name: "a"
        values: [2]
*)
(* @assert:cpu:error
stage: frontend
message: "duplicate tensor argument name"
*)
(* @assert:metal:error
stage: frontend
message: "duplicate tensor argument name"
*)
a

(* @case invalid-second-name *)
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: tensor-input-cpu
    arguments:
      - name: "a"
        values: [1]
      - name: "2b"
        values: [2]
  - name: metal
    pipeline: tensor-input-metal
    threads-per-group: 4
    arguments:
      - name: "a"
        values: [1]
      - name: "2b"
        values: [2]
*)
(* @assert:cpu:error
stage: frontend
message: "invalid tensor argument name"
*)
(* @assert:metal:error
stage: frontend
message: "invalid tensor argument name"
*)
a

(* @case reserved-third-name *)
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: tensor-input-cpu
    arguments:
      - name: "a"
        values: [1]
      - name: "b"
        values: [2]
      - name: "let"
        values: [3]
  - name: metal
    pipeline: tensor-input-metal
    threads-per-group: 4
    arguments:
      - name: "a"
        values: [1]
      - name: "b"
        values: [2]
      - name: "let"
        values: [3]
*)
(* @assert:cpu:error
stage: frontend
message: "invalid tensor argument name"
*)
(* @assert:metal:error
stage: frontend
message: "invalid tensor argument name"
*)
a

(* @case unknown *)
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: tensor-input-cpu
    arguments:
      - name: "a"
        values: []
      - name: "b"
        values: []
  - name: metal
    pipeline: tensor-input-metal
    threads-per-group: 4
    arguments:
      - name: "a"
        values: []
      - name: "b"
        values: []
*)
(* @assert:cpu:error
stage: frontend
message: "unknown tensor expression variable"
*)
(* @assert:metal:error
stage: frontend
message: "unknown tensor expression variable"
*)
a + missing
