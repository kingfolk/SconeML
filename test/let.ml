(*
@letalg:opt
module {
  func.func @test_function() {
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %0 = arith.addi %c1_i32, %c2_i32 : i32
    %1 = "letalg.yield"(%0) : (i32) -> i32
  }
}
*)
let x = 1 in let y = 2 in x + y
