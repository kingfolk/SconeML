// @test
// version: 1
// input: mlir
// runs:
//   - name: lower
//     pipeline: tensor-gpu-ir
// @end
// @assert:lower:error
// stage: gpu-lowering
// op: letalg.tensor_map
// message: identity layout in the default host memory space
// @end
module {
  func.func @map_source(%input: memref<4xi32, 3>, %output: memref<4xi32, 3>) {
    "letalg.tensor_map"(%input, %output) ({
    ^bb0(%x: i32):
      %square = arith.muli %x, %x : i32
      %twice = arith.addi %square, %square : i32
      letalg.yield %twice : i32
    }) : (memref<4xi32, 3>, memref<4xi32, 3>) -> ()
    func.return
  }
}
