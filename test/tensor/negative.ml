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
  - name: metal
    pipeline: tensor-metal-legacy
*)
(* @assert:cpu:result
[10, -2, -11, -23]
*)
(* @assert:kernel:ir-check
CHECK: gpu.func @tensor_map
CHECK: scf.if
CHECK: memref.load
CHECK: memref.store
CHECK-NOT: letalg.tensor_map
*)
(* @assert:kernel:source-unchanged
true
*)
(* @assert:shader:text-check
CHECK: kernel void tensor_map
CHECK: const uint
CHECK: output[
*)
(* @assert:shader:source-unchanged
true
*)
(* @assert:cpu:ir-check
CHECK: llvm.func @tensor_literal
*)
(* @assert:metal:text-check
CHECK: kernel void tensor_map
*)
(* @assert:metal:compare
cpu
*)
let xs = [0, 4, 7, 11] in 10 - xs * 3
