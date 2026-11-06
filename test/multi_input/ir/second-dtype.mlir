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
// message: matching rank-1
// @end
// @assert:gpu:error
// stage: gpu-lowering
// message: matching rank-1
// @end
// @assert:cpu:error
// stage: cpu-lowering
// message: matching rank-1
// @end
module {
  func.func @map_source(%a: memref<4xi32>, %b: memref<4xf32>, %c: memref<4xi32>, %out: memref<4xi32>) {
    "letalg.tensor_map"(%a, %b, %c, %out) ({
    ^bb0(%x: i32, %y: i32, %z: i32):
      %prod = arith.muli %x, %y : i32
      %sum = arith.addi %prod, %z : i32
      letalg.yield %sum : i32
    }) : (memref<4xi32>, memref<4xf32>, memref<4xi32>, memref<4xi32>) -> ()
    func.return
  }
}
