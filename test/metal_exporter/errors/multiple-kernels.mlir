// @test
// version: 1
// input: mlir
// runs:
//   - name: export
//     pipeline: tensor-msl
// @end
// @assert:export:error
// stage: msl-export
// op: gpu.module
// message: exactly one defined GPU kernel
// @end
module {
  gpu.module @kernels {
    gpu.func @other() kernel { gpu.return }
    gpu.func @tensor_map(%input: memref<?xi32, #gpu.address_space<global>>,
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
        %square = arith.muli %x, %x : i32
        %twice = arith.addi %square, %square : i32
        memref.store %twice, %output[%index] : memref<?xi32, #gpu.address_space<global>>
      }
      gpu.return
    }
  }
}
