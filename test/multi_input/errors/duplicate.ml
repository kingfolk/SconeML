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
