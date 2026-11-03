// Host-side source: the scalar square is shared by both add operands.
module {
  func.func @map_source(%input: memref<4xi32>, %output: memref<4xi32>) {
    "letalg.tensor_map"(%input, %output) ({
    ^bb0(%x: i32):
      %square = arith.muli %x, %x : i32
      %twice = arith.addi %square, %square : i32
      letalg.yield %twice : i32
    }) : (memref<4xi32>, memref<4xi32>) -> ()
    func.return
  }
}
