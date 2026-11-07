(* @case invalid-width *)
(* @test
version: 1
input: source
runs:
  - name: nonempty
    pipeline: tensor-input-metal
    threads-per-group: 4294967296
    arguments:
      - name: xs
        values: [1]
  - name: empty
    pipeline: tensor-input-metal
    threads-per-group: 4294967296
    arguments:
      - name: xs
        values: []
*)
(* @assert:nonempty:error
stage: metal-execute
message: threads per threadgroup exceeds supported range
*)
(* @assert:empty:error
stage: metal-execute
message: threads per threadgroup exceeds supported range
*)
xs + 1
