// @test
// version: 1
// input: mlir
// runs:
//   - name: effects
//     pipeline: tensor-effects
//   - name: opt
//     pipeline: letalg-opt
//   - name: loops
//     pipeline: tensor-scf
// @end
// @assert:effects:effects
// map 0:
//   read operand 0
//   write operand 1
// map 1:
//   read operand 0
//   write operand 1
// @end
// @assert:opt:ir-check
// CHECK: "letalg.tensor_map"
// CHECK: "letalg.tensor_map"
// CHECK-NOT: "letalg.tensor_map"
// @end
// @assert:loops:ir-check
// CHECK-NOT: letalg.tensor_map
// CHECK: scf.for
// CHECK: memref.load
// CHECK: memref.store
// CHECK: scf.for
// CHECK: memref.load
// CHECK: memref.store
// CHECK-NOT: scf.for
// CHECK-NOT: letalg.tensor_map
// @end
module {
  func.func @test(%in: memref<4xi32>, %out: memref<4xi32>, %capture: i32) {
    "letalg.tensor_map"(%in, %out) ({
    ^bb0(%x: i32):
      %c = arith.constant 1 : i32
      %v = arith.addi %x, %c : i32
      letalg.yield %v : i32
    }) : (memref<4xi32>, memref<4xi32>) -> ()
    "letalg.tensor_map"(%in, %out) ({
    ^bb0(%x: i32):
      %c = arith.constant 1 : i32
      %v = arith.addi %x, %c : i32
      letalg.yield %v : i32
    }) : (memref<4xi32>, memref<4xi32>) -> ()
    func.return
  }
}
