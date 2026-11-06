// @test
// version: 1
// input: mlir
// runs:
//   - name: export
//     pipeline: tensor-msl
// @end
// @assert:export:error
// stage: msl-export
// message: expected input[index] load
// @end
module {
  gpu.module @kernels {
    gpu.func @wrong_load_order(%a: memref<?xi32, #gpu.address_space<global>>, %b: memref<?xi32, #gpu.address_space<global>>, %out: memref<?xi32, #gpu.address_space<global>>, %count: i32) kernel {
      %block = gpu.block_id x
      %width = gpu.block_dim x
      %thread = gpu.thread_id x
      %base = arith.muli %block, %width : index
      %index = arith.addi %base, %thread : index
      %size = arith.index_castui %count : i32 to index
      %valid = arith.cmpi ult, %index, %size : index
      scf.if %valid {
        %x = memref.load %b[%index] : memref<?xi32, #gpu.address_space<global>>
        %y = memref.load %a[%index] : memref<?xi32, #gpu.address_space<global>>
        %result = arith.subi %x, %y : i32
        memref.store %result, %out[%index] : memref<?xi32, #gpu.address_space<global>>
      }
      gpu.return
    }
  }
}
