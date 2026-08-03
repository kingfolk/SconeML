(*
@letalg:tensor
func.func @tensor_literal
memref<3xi32>
memref.alloc
memref.store
letalg.tensor_map
arith.muli
arith.addi
letalg.yield
*)
let ts = [1, 2, 3] in ts * ts + 2 * ts + 1
