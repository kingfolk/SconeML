(*
@letalg:tensor
func.func @polynomial2
memref<?xf32>
letalg.tensor_map
arith.mulf
arith.subf
arith.addf
letalg.yield
*)
let polynomial2 (x : tensor<f32>) = x * x - 3.0 * x + 2.0
