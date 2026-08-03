(*
@letalg:opt
module {
  func.func @test_function() {
    %alloc = memref.alloc() : memref<3xi32>
    %c0 = arith.constant 0 : index
    %c1_i32 = arith.constant 1 : i32
    memref.store %c1_i32, %alloc[%c0] : memref<3xi32>
    %c1 = arith.constant 1 : index
    %c2_i32 = arith.constant 2 : i32
    memref.store %c2_i32, %alloc[%c1] : memref<3xi32>
    %c2 = arith.constant 2 : index
    %c3_i32 = arith.constant 3 : i32
    memref.store %c3_i32, %alloc[%c2] : memref<3xi32>
    %alloc_0 = memref.alloc() : memref<3xi32>
    "letalg.tensor_map"(%alloc, %alloc_0) ({
    ^bb0(%arg0: i32):
      %0 = arith.muli %arg0, %arg0 : i32
      %c2_i32_1 = arith.constant 2 : i32
      %1 = arith.muli %c2_i32_1, %arg0 : i32
      %2 = arith.addi %0, %1 : i32
      %c1_i32_2 = arith.constant 1 : i32
      %3 = arith.addi %2, %c1_i32_2 : i32
      letalg.yield %3 : i32
    }) : (memref<3xi32>, memref<3xi32>) -> ()
    letalg.yield %alloc_0 : memref<3xi32>
  }
}
*)
let ts = [1, 2, 3] in ts * ts + 2 * ts + 1
