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
