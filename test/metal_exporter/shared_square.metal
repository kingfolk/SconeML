#include <metal_stdlib>
using namespace metal;

kernel void tensor_map(device const uint *input [[buffer(0)]],
                       device uint *output [[buffer(1)]],
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
    const uint v7 = input[v4];
    const uint v8 = v7 * v7;
    const uint v9 = v8 + v8;
    output[v4] = v9;
  }
}
