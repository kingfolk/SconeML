(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: tensor-cpu
  - name: kernel
    pipeline: tensor-gpu-ir
  - name: shader
    pipeline: tensor-msl
*)
(* @assert:cpu:result
[1]
*)
(* @assert:kernel:ir-check
CHECK: scf.if
CHECK: %[[E:.*]] = memref.load
CHECK-NEXT: memref.store %[[E]]
*)
(* @assert:shader:text-check
CHECK: input[
CHECK-NEXT: output[
*)
let xs = [1] in xs
