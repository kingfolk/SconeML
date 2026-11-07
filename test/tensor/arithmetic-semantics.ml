(* @case identity *)
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
    threads-per-group: 4
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
(* @assert:metal:compare
cpu
*)
let xs = [1] in xs

(* @case negative *)
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
CHECK: threadgroup_position_in_grid
CHECK: threads_per_threadgroup
CHECK: thread_position_in_threadgroup
CHECK: const uint v0 = group_id.x;
*)
(* @assert:metal:compare
cpu
*)
let xs = [0, 4, 7, 11] in 10 - xs * 3

(* @case overflow *)
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
