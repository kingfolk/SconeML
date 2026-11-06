// @test
// version: 1
// input: mlir
// runs:
//   - name: kernel
//     pipeline: tensor-gpu-ir
// @end
// @assert:kernel:ir-check
// CHECK: %[[A:.*]] = memref.load
// CHECK-NEXT: %[[B:.*]] = memref.load
// CHECK-NEXT: %[[C:.*]] = memref.load
// CHECK-NEXT: %[[P:.*]] = arith.muli %[[A]], %[[B]] : i32
// CHECK-NEXT: %[[S:.*]] = arith.addi %[[P]], %[[P]] : i32
// CHECK-NEXT: memref.store %[[S]]
// @end
module {
  func.func @map_source(%a: memref<4xi32>, %b: memref<4xi32>, %c: memref<4xi32>, %out: memref<4xi32>) {
    "letalg.tensor_map"(%a, %b, %c, %out) ({
    ^bb0(%x: i32, %y: i32, %z: i32):
      %prod = arith.muli %x, %y : i32
      %sum = arith.addi %prod, %prod : i32
      letalg.yield %sum : i32
    }) : (memref<4xi32>, memref<4xi32>, memref<4xi32>, memref<4xi32>) -> ()
    func.return
  }
}
