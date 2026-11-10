(* @case row-broadcast *)
(* @test
version: 1
input: source
runs:
  - name: device
    pipeline: shaped-f32-msl
    profile: f32-ftz
    arguments:
      - name: a
        dtype: f32
        values: [1, 2, 3, 4, 5, 6]
        shape: [2, 3]
      - name: b
        dtype: f32
        values: [0.25, -0.5, 1.5]
        shape: [3]
*)
(* @assert:device:text-check
CHECK: #pragma STDC FP_CONTRACT OFF
CHECK: device const float *input0
CHECK: device const float *input1
CHECK: device float *output
CHECK: constant uint &count
CHECK: const uint
CHECK: const float
CHECK: const ulong
CHECK: 140737488355328ul
*)
(* @assert:device:ir-check
CHECK: sconeml.element_bytes = 4 : i64
CHECK: sconeml.element_type = f32
CHECK: sconeml.float_profile = "f32-ftz"
CHECK: arith.mulf
CHECK-NOT: fastmath<
*)
(* @assert:device:entrypoint
tensor_map
*)
(* @assert:device:source-unchanged
true
*)
a * 0.5 + b

