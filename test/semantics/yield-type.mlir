// @test
// version: 1
// input: mlir
// runs:
//   - name: verify
//     pipeline: verify
//   - name: lower
//     pipeline: tensor-gpu-ir
//   - name: cpu-lower
//     pipeline: tensor-scf
// @end
// @assert:verify:error
// stage: verify
// op: letalg.tensor_map
// message: body must yield
// @end
// @assert:lower:error
// stage: gpu-lowering
// op: letalg.tensor_map
// message: body must yield
// @end
// @assert:cpu-lower:error
// stage: cpu-lowering
// op: letalg.tensor_map
// message: body must yield
// @end
module {
  func.func @test(%in: memref<4xi32>, %out: memref<4xi32>, %capture: i32) {
    "letalg.tensor_map"(%in, %out) ({
    ^bb0(%x: i32):
      %c = arith.constant 1 : i64
      letalg.yield %c : i64
    }) : (memref<4xi32>, memref<4xi32>) -> ()
    func.return
  }
}
