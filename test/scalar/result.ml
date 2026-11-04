(* @test
version: 1
input: source
runs:
  - name: initial
    pipeline: letalg
  - name: cpu
    pipeline: scalar-cpu
*)
(* @assert:initial:ir-check
CHECK: func.func @tensor_literal
CHECK: letalg.let
CHECK: arith.addi
*)
(* @assert:cpu:result
3
*)
let x = 1 in let y = 2 in x + y
