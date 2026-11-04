// @test
// version: 1
// input: mlir
// runs:
//   - name: export
//     pipeline: tensor-msl
// @end
// @assert:export:error
// stage: msl-export
// op: builtin.module
// message: exactly one gpu.module
// @end
module {}
