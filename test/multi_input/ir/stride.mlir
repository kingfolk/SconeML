// @test
// version: 1
// input: mlir
// runs:
//   - name: gpu
//     pipeline: tensor-gpu-ir
// @end
// @assert:gpu:error
// stage: gpu-lowering
// message: identity layout
// @end
module {
  func.func @map_source(%a: memref<4xi32, strided<[2], offset: 0>>, %b: memref<4xi32, strided<[2], offset: 0>>, %c: memref<4xi32, strided<[2], offset: 0>>, %out: memref<4xi32, strided<[2], offset: 0>>) {
    "letalg.tensor_map"(%a, %b, %c, %out) ({
    ^bb0(%x: i32, %y: i32, %z: i32):
      %prod = arith.muli %x, %y : i32
      %sum = arith.addi %prod, %z : i32
      letalg.yield %sum : i32
    }) : (memref<4xi32, strided<[2], offset: 0>>, memref<4xi32, strided<[2], offset: 0>>, memref<4xi32, strided<[2], offset: 0>>, memref<4xi32, strided<[2], offset: 0>>) -> ()
    func.return
  }
}
