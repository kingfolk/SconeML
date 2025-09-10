(*
@letalg:opt
module {
  func.func @test_function() {
    %c1_i32 = arith.constant 1 : i32
    %0 = letalg.lambda "f" (%arg0: i32,%arg1: i32){
      %3 = arith.subi %arg1, %arg0 : i32
      %c10_i32 = arith.constant 10 : i32
      %4 = arith.addi %3, %c10_i32 : i32
      %5 = "letalg.yield"(%4) : (i32) -> i32
    } -> (i32) -> i32
    %c2_i32 = arith.constant 2 : i32
    %1 = "letalg.apply"(%0, %c1_i32, %c2_i32) : ((i32) -> i32, i32, i32) -> i32
    %2 = "letalg.yield"(%1) : (i32) -> i32
  }
}
*)
let a = 1 in let f x = x - a + 10 in f 2