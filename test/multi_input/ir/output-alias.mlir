// @test
// version: 1
// input: mlir
// runs:
//   - name: verify
//     pipeline: verify
//   - name: gpu
//     pipeline: tensor-gpu-ir
//   - name: cpu
//     pipeline: tensor-scf
// @end
// @assert:verify:error
// stage: verify
// message: distinct input and output
// @end
// @assert:gpu:error
// stage: gpu-lowering
// message: distinct input and output
// @end
// @assert:cpu:error
// stage: cpu-lowering
// message: distinct input and output
// @end
module {
  func.func @map_source(%a: memref<4xi32>, %b: memref<4xi32>, %c: memref<4xi32>, %out: memref<4xi32>) {
    "letalg.tensor_map"(%a, %b, %c, %b) ({
    ^bb0(%x: i32, %y: i32, %z: i32):
      %prod = arith.muli %x, %y : i32
      %sum = arith.addi %prod, %z : i32
      letalg.yield %sum : i32
    }) : (memref<4xi32>, memref<4xi32>, memref<4xi32>, memref<4xi32>) -> ()
    func.return
  }
}
