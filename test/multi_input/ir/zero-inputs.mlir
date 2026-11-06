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
// message: between one and three input buffers
// @end
// @assert:gpu:error
// stage: gpu-lowering
// message: between one and three input buffers
// @end
// @assert:cpu:error
// stage: cpu-lowering
// message: between one and three input buffers
// @end
module {
  func.func @zero(%out: memref<4xi32>) {
    "letalg.tensor_map"(%out) ({
      %zero = arith.constant 0 : i32
      letalg.yield %zero : i32
    }) : (memref<4xi32>) -> ()
    func.return
  }
}
