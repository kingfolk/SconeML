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
