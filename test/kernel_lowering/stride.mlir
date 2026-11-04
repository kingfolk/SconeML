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
  func.func @map_source(%input: memref<4xi32, strided<[2], offset: 1>>, %output: memref<4xi32, strided<[2], offset: 1>>) {
    "letalg.tensor_map"(%input, %output) ({
    ^bb0(%x: i32):
      %square = arith.muli %x, %x : i32
      %twice = arith.addi %square, %square : i32
      letalg.yield %twice : i32
    }) : (memref<4xi32, strided<[2], offset: 1>>, memref<4xi32, strided<[2], offset: 1>>) -> ()
    func.return
  }
}
