(*
@letalg:opt
module {
  func.func @test_function() {
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c10_i32 = arith.constant 10 : i32
    %0 = arith.addi %c2_i32, %c10_i32 : i32
    %1 = arith.addi %c1_i32, %0 : i32
    %2 = "letalg.yield"(%1) : (i32) -> i32
  }
}
*)
let x = 1 in x + let y = 2 in y + 10