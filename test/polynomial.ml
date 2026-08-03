(*
@tensor:ir
func.func @tensor_polynomial
memref<?xf32>
letalg.tensor_map
arith.mulf
arith.addf
letalg.yield
*)
let polynomial x = x * x + 2.0 * x + 1.0
