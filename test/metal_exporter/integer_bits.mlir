// @test
// version: 1
// input: mlir
// runs:
//   - name: shader
//     pipeline: tensor-msl
//   - name: compile
//     pipeline: metal-compile
// @end
// @assert:shader:text-equal
// #include <metal_stdlib>
// using namespace metal;
//
// kernel void tensor_map(device const uint *input [[buffer(0)]],
//                        device uint *output [[buffer(1)]],
//                        constant uint &count [[buffer(2)]],
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
//     const uint v7 = input[v4];
//     const uint v8 = 4294967295u;
//     const uint v9 = 2147483648u;
//     const uint v10 = 2147483647u;
//     const uint v11 = 0u;
//     const uint v12 = v7 + v10;
//     const uint v13 = v12 * v9;
//     const uint v14 = v13 - v8;
//     const uint v15 = v14 + v11;
//     output[v4] = v15;
//   }
// }
// @end
// @assert:compile:success
// true
// @end
// @assert:shader:entrypoint
// tensor_map
// @end
// @assert:shader:source-unchanged
// true
// @end
module {
  gpu.module @kernels {
    gpu.func @integer_bits(%input: memref<?xi32, #gpu.address_space<global>>,
                           %output: memref<?xi32, #gpu.address_space<global>>,
                           %count: i32) kernel {
      %block = gpu.block_id x
      %width = gpu.block_dim x
      %thread = gpu.thread_id x
      %base = arith.muli %block, %width : index
      %index = arith.addi %base, %thread : index
      %size = arith.index_castui %count : i32 to index
      %in_bounds = arith.cmpi ult, %index, %size : index
      scf.if %in_bounds {
        %x = memref.load %input[%index] : memref<?xi32, #gpu.address_space<global>>
        %neg = arith.constant -1 : i32
        %min = arith.constant -2147483648 : i32
        %max = arith.constant 2147483647 : i32
        %zero = arith.constant 0 : i32
        %sum = arith.addi %x, %max : i32
        %prod = arith.muli %sum, %min : i32
        %diff = arith.subi %prod, %neg : i32
        %result = arith.addi %diff, %zero : i32
        memref.store %result, %output[%index] : memref<?xi32, #gpu.address_space<global>>
      }
      gpu.return
    }
  }
}
