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
