// @test
// version: 1
// input: mlir
// runs:
//   - name: effects
//     pipeline: tensor-effects
// @end
// @assert:effects:effects
// map 0:
//   read operand 0
//   read operand 1
//   read operand 2
//   write operand 3
// @end
module {
  func.func @map_source(%a: memref<4xi32>, %b: memref<4xi32>, %c: memref<4xi32>, %out: memref<4xi32>) {
    "letalg.tensor_map"(%a, %a, %c, %out) ({
    ^bb0(%x: i32, %y: i32, %z: i32):
      %prod = arith.muli %x, %y : i32
      %sum = arith.addi %prod, %z : i32
      letalg.yield %sum : i32
    }) : (memref<4xi32>, memref<4xi32>, memref<4xi32>, memref<4xi32>) -> ()
    func.return
  }
}
