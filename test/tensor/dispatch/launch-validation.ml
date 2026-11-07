(* @case invalid-width *)
(* @test
version: 1
input: source
runs:
  - name: rejected
    pipeline: tensor-metal
    threads-per-group: 4294967296
*)
(* @assert:rejected:error
stage: metal-execute
message: threads per threadgroup exceeds supported range
*)
let xs = [1] in xs + 1
