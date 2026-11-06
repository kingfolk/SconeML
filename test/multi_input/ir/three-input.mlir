// @test
// version: 1
// input: mlir
// runs:
//   - name: effects
//     pipeline: tensor-effects
//   - name: cpu
//     pipeline: tensor-scf
//   - name: kernel
//     pipeline: tensor-gpu-ir
//   - name: shader
//     pipeline: tensor-msl
// @end
// @assert:effects:effects
// map 0:
//   read operand 0
//   read operand 1
//   read operand 2
//   write operand 3
// @end
// @assert:cpu:ir-check
// CHECK: scf.for
// CHECK: %[[A:.*]] = memref.load
// CHECK-NEXT: %[[B:.*]] = memref.load
// CHECK-NEXT: %[[C:.*]] = memref.load
// CHECK-NEXT: %[[P:.*]] = arith.muli %[[A]], %[[B]] : i32
// CHECK-NEXT: %[[S:.*]] = arith.addi %[[P]], %[[C]] : i32
// CHECK-NEXT: memref.store %[[S]]
// @end
// @assert:kernel:ir-check
// CHECK: %[[A:.*]] = memref.load
// CHECK-NEXT: %[[B:.*]] = memref.load
// CHECK-NEXT: %[[C:.*]] = memref.load
// CHECK-NEXT: %[[P:.*]] = arith.muli %[[A]], %[[B]] : i32
// CHECK-NEXT: %[[S:.*]] = arith.addi %[[P]], %[[C]] : i32
// CHECK-NEXT: memref.store %[[S]]
// @end
// @assert:shader:text-equal
// #include <metal_stdlib>
// using namespace metal;
//
// kernel void tensor_map(device const uint *input0 [[buffer(0)]],
//                        device const uint *input1 [[buffer(1)]],
//                        device const uint *input2 [[buffer(2)]],
//                        device uint *output [[buffer(3)]],
//                        constant uint &count [[buffer(4)]],
//                        uint3 group_id [[threadgroup_position_in_grid]],
//                        uint3 group_size [[threads_per_threadgroup]],
//                        uint3 local_id [[thread_position_in_threadgroup]]) {
//   const uint v0 = group_id.x;
//   const uint v1 = group_size.x;
//   const uint v2 = local_id.x;
//   const uint v3 = v0 * v1;
//   const uint v4 = v3 + v2;
//   const uint v5 = count;
//   const bool v6 = v4 < v5;
//   if (v6) {
//     const uint v7 = input0[v4];
//     const uint v8 = input1[v4];
//     const uint v9 = input2[v4];
//     const uint v10 = v7 * v8;
//     const uint v11 = v10 + v9;
//     output[v4] = v11;
//   }
// }
// @end
// @assert:shader:source-unchanged
// true
// @end
module {
  func.func @map_source(%a: memref<4xi32>, %b: memref<4xi32>, %c: memref<4xi32>, %out: memref<4xi32>) {
    "letalg.tensor_map"(%a, %b, %c, %out) ({
    ^bb0(%x: i32, %y: i32, %z: i32):
      %prod = arith.muli %x, %y : i32
      %sum = arith.addi %prod, %z : i32
      letalg.yield %sum : i32
    }) : (memref<4xi32>, memref<4xi32>, memref<4xi32>, memref<4xi32>) -> ()
    func.return
  }
}
