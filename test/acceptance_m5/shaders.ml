(* @case full-i32-shader *)
(* @test
version: 1
input: source
runs:
  - name: shader
    pipeline: sum-i32-msl
    arguments:
      - name: a
        dtype: i32
        values: [1,2,3,4,5,6]
        shape: [2,3]
*)
(* @assert:shader:text-equal
#include <metal_stdlib>
using namespace metal;
#pragma STDC FP_CONTRACT OFF
using Element = uint;
inline uint sum_normalize(uint v) { return v; }
inline uint sum_add(uint a, uint b) { return a + b; }
kernel void sum_tree(device const Element *input [[buffer(0)]],
                     device Element *output [[buffer(1)]],
                     constant uint &length [[buffer(2)]],
                     constant uint &parts [[buffer(3)]],
                     constant uint &base_stride [[buffer(4)]],
                     constant uint &input_stride [[buffer(5)]],
                     uint group [[threadgroup_position_in_grid]],
                     uint lane [[thread_index_in_threadgroup]]) {
  threadgroup Element scratch[256];
  const uint row = group / parts;
  const uint chunk = group % parts;
  const uint k = chunk * 256u + lane;
  scratch[lane] = k < length ? sum_normalize(input[row * base_stride + k * input_stride]) : Element(0);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = 1u; stride < 256u; stride *= 2u) {
    if (lane % (2u * stride) == 0u)
      scratch[lane] = sum_add(scratch[lane], scratch[lane + stride]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lane == 0u) output[group] = scratch[0];
}
*)
(* @assert:shader:entrypoint
sum_tree
*)
sum(a, axis=1)

(* @case full-f32-shader *)
(* @test
version: 1
input: source
runs:
  - name: shader
    pipeline: sum-f32-msl
    profile: f32-sum-tree256-ftz
    arguments:
      - name: a
        dtype: f32
        values: [1,2,3,4,5,6]
        shape: [2,3]
*)
(* @assert:shader:text-equal
#include <metal_stdlib>
using namespace metal;
#pragma STDC FP_CONTRACT OFF
using Element = float;
inline float sum_normalize(float v) {
  uint bits = as_type<uint>(v);
  return as_type<float>((bits & 0x7fffffffu) < 0x00800000u ? 0u : bits);
}
inline float sum_add(float a, float b) { return sum_normalize(a + b); }
kernel void sum_tree(device const Element *input [[buffer(0)]],
                     device Element *output [[buffer(1)]],
                     constant uint &length [[buffer(2)]],
                     constant uint &parts [[buffer(3)]],
                     constant uint &base_stride [[buffer(4)]],
                     constant uint &input_stride [[buffer(5)]],
                     uint group [[threadgroup_position_in_grid]],
                     uint lane [[thread_index_in_threadgroup]]) {
  threadgroup Element scratch[256];
  const uint row = group / parts;
  const uint chunk = group % parts;
  const uint k = chunk * 256u + lane;
  scratch[lane] = k < length ? sum_normalize(input[row * base_stride + k * input_stride]) : Element(0);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = 1u; stride < 256u; stride *= 2u) {
    if (lane % (2u * stride) == 0u)
      scratch[lane] = sum_add(scratch[lane], scratch[lane + stride]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lane == 0u) output[group] = scratch[0];
}
*)
(* @assert:shader:entrypoint
sum_tree
*)
sum(a, axis=1)

