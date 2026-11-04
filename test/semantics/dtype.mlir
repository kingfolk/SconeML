// @test
// version: 1
// input: mlir
// runs:
//   - name: verify
//     pipeline: verify
//   - name: lower
//     pipeline: tensor-gpu-ir
// @end
// @assert:verify:error
// stage: verify
// op: letalg.tensor_map
// message: i32 input and output
// @end
// @assert:lower:error
// stage: gpu-lowering
// op: letalg.tensor_map
// message: i32 input and output
// @end
module {
  func.func @test(%in: memref<4xf32>, %out: memref<4xf32>, %capture: i32) {
    "letalg.tensor_map"(%in, %out) ({
    ^bb0(%x: f32):
      letalg.yield %x : f32
    }) : (memref<4xf32>, memref<4xf32>) -> ()
    func.return
  }
}
