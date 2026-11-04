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
// message: matching rank-1
// @end
// @assert:lower:error
// stage: gpu-lowering
// op: letalg.tensor_map
// message: matching rank-1
// @end
module {
  func.func @test(%in: memref<2x2xi32>, %out: memref<2x2xi32>, %capture: i32) {
    "letalg.tensor_map"(%in, %out) ({
    ^bb0(%x: i32):
      letalg.yield %x : i32
    }) : (memref<2x2xi32>, memref<2x2xi32>) -> ()
    func.return
  }
}
