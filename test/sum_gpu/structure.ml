(* @test
version: 1
input: source
runs:
  - name: ir
    pipeline: sum-i32-msl
    arguments:
      - name: a
        values: [1,2,3,4,5,6]
        shape: [2,3]
*)
(* @assert:ir:text-check
CHECK: threadgroup Element scratch[256]
CHECK: k < length ?
CHECK: threadgroup_barrier(mem_flags::mem_threadgroup)
CHECK: stride *= 2u
CHECK: scratch[lane] = sum_add(scratch[lane], scratch[lane + stride])
CHECK: threadgroup_barrier(mem_flags::mem_threadgroup)
CHECK: output[group] = scratch[0]
*)
sum(a, axis=1)
