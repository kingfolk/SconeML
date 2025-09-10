(*
@letalg:opt
module {
  func.func @test_function() {
    %0 = letalg.lambda "f" (%arg0: i32){
      %c10_i32 = arith.constant 10 : i32
      %3 = arith.addi %arg0, %c10_i32 : i32
      %4 = "letalg.yield"(%3) : (i32) -> i32
    } -> (i32) -> i32
    %c2_i32 = arith.constant 2 : i32
    %1 = "letalg.apply"(%0, %c2_i32) : ((i32) -> i32, i32) -> i32
    %2 = "letalg.yield"(%1) : (i32) -> i32
  }
}
*)
let f x = x + 10 in f 2