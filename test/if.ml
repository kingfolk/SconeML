(*
@letalg:opt
module {
  func.func @test_function() {
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = arith.cmpi eq, %c1_i32, %c0_i32 : i32
    %1 = scf.if %0 -> (i32) {
      %c10_i32 = arith.constant 10 : i32
      %3 = arith.addi %c1_i32, %c10_i32 : i32
      scf.yield %3 : i32
    } else {
      %c0_i32_0 = arith.constant 0 : i32
      scf.yield %c0_i32_0 : i32
    }
    %2 = "letalg.yield"(%1) : (i32) -> i32
  }
}
*)
let x = 1 in if x then x + 10 else 0