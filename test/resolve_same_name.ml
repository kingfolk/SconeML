(*
@letalg:opt
module {
  func.func @test_function() {
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %0 = arith.subi %c1_i32, %c2_i32 : i32
    %1 = "letalg.yield"(%0) : (i32) -> i32
  }
}
*)
let x = 1 in x - let x = 2 in x