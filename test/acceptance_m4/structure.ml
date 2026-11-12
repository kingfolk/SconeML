(* @case f32-add-golden *)
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: shaped-f32-cpu
    profile: f32-ftz
    arguments:
      - name: a
        dtype: f32
        shape: [3]
        values: [0x3f800000, 0x3f000000, 0x80000000]
  - name: metal
    pipeline: shaped-f32-metal
    profile: f32-ftz
    threads-per-group: 4
    arguments:
      - name: a
        dtype: f32
        shape: [3]
        values: [0x3f800000, 0x3f000000, 0x80000000]
  - name: device
    pipeline: shaped-f32-msl
    profile: f32-ftz
    arguments:
      - name: a
        dtype: f32
        shape: [3]
        values: [1.0, 0.5, -0.0]
*)
(* @assert:cpu:result
[0x3fc00000, 0x3f800000, 0x3f000000]
*)
(* @assert:cpu:shape
[3]
*)
(* @assert:metal:result
[0x3fc00000, 0x3f800000, 0x3f000000]
*)
(* @assert:metal:shape
[3]
*)
(* @assert:metal:compare
cpu
*)
(* @assert:device:text-equal
#include <metal_stdlib>
using namespace metal;

#pragma STDC FP_CONTRACT OFF

kernel void tensor_map(device const float *input [[buffer(0)]],
                       device float *output [[buffer(1)]],
                       constant uint &count [[buffer(2)]],
                       uint3 group_id [[threadgroup_position_in_grid]],
                       uint3 group_size [[threads_per_threadgroup]],
                       uint3 local_id [[thread_position_in_threadgroup]]) {
  const uint v0 = group_id.x;
  const uint v1 = group_size.x;
  const uint v2 = local_id.x;
  const uint v3 = v0 * v1;
  const uint v4 = v3 + v2;
  const uint v5 = count;
  const bool v6 = v4 < v5;
  if (v6) {
    const float v7 = input[v4];
    const uint v8 = 2147483647u;
    const uint v9 = 8388608u;
    const uint v10 = 2147483648u;
    const uint v11 = as_type<uint>(v7);
    const uint v12 = v11 & v8;
    const bool v13 = v12 < v9;
    const uint v14 = v11 & v10;
    const uint v15 = v13 ? v14 : v11;
    const float v16 = as_type<float>(v15);
    const float v17 = as_type<float>(1056964608u);
    const float v18 = v16 + v17;
    const uint v19 = 2147483647u;
    const uint v20 = 8388608u;
    const uint v21 = 2147483648u;
    const uint v22 = as_type<uint>(v18);
    const uint v23 = v22 & v19;
    const bool v24 = v23 < v20;
    const uint v25 = v22 & v21;
    const uint v26 = v24 ? v25 : v22;
    const float v27 = as_type<float>(v26);
    output[v4] = v27;
  }
}
*)
(* @assert:device:source-unchanged
true
*)
(* @assert:device:entrypoint
tensor_map
*)
a + 0.5

