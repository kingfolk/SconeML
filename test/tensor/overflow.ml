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
    pipeline: tensor-metal
*)
(* @assert:cpu:result
[-1794967296, 1]
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
CHECK: threadgroup_position_in_grid
CHECK: threads_per_threadgroup
CHECK: thread_position_in_threadgroup
CHECK: const uint v0 = group_id.x;
*)
(* @assert:metal:compare
cpu
*)
let xs = [50000, 1] in xs * xs
