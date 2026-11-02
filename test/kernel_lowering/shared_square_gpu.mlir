// Expected device-only extraction. No host literal/alloc/store is required.
module {
  gpu.module @kernels {
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
