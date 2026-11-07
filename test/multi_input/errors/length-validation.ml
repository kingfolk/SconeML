(* @case mismatched *)
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: tensor-input-cpu
    arguments:
      - name: "a"
        values: [1, 2]
      - name: "b"
        values: [3]
  - name: metal
    pipeline: tensor-input-metal
    threads-per-group: 4
    arguments:
      - name: "a"
        values: [1, 2]
      - name: "b"
        values: [3]
*)
(* @assert:cpu:error
stage: frontend
message: "matching lengths"
*)
(* @assert:metal:error
stage: frontend
message: "matching lengths"
*)
a + b

(* @case empty-mismatched *)
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
        values: [3]
  - name: metal
    pipeline: tensor-input-metal
    threads-per-group: 4
    arguments:
      - name: "a"
        values: []
      - name: "b"
        values: [3]
*)
(* @assert:cpu:error
stage: frontend
message: "matching lengths"
*)
(* @assert:metal:error
stage: frontend
message: "matching lengths"
*)
a + b

(* @case unused-mismatched *)
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: tensor-input-cpu
    arguments:
      - name: "a"
        values: [1, 2]
      - name: "b"
        values: [3]
  - name: metal
    pipeline: tensor-input-metal
    threads-per-group: 4
    arguments:
      - name: "a"
        values: [1, 2]
      - name: "b"
        values: [3]
*)
(* @assert:cpu:error
stage: frontend
message: "matching lengths"
*)
(* @assert:metal:error
stage: frontend
message: "matching lengths"
*)
a
