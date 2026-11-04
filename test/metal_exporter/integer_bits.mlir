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
