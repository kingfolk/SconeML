(*
@letalg:tensor
func.func @polynomial
memref<?xf32>
letalg.tensor_map
arith.mulf
arith.addf
letalg.yield
*)
let polynomial (x : tensor<f32>) = x * x + 2.0 * x + 1.0
