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
