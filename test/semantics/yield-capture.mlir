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
// message: must not capture
// @end
// @assert:lower:error
// stage: gpu-lowering
// op: letalg.tensor_map
// message: must not capture
// @end
module {
  func.func @test(%in: memref<4xi32>, %out: memref<4xi32>, %capture: i32) {
    "letalg.tensor_map"(%in, %out) ({
    ^bb0(%x: i32):
      letalg.yield %capture : i32
    }) : (memref<4xi32>, memref<4xi32>) -> ()
    func.return
  }
}
