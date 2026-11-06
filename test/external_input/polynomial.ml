(* @test
version: 1
input: source
runs:
  - name: first
    pipeline: tensor-input-cpu
    arguments:
      - name: "xs"
        values: [-2, 0, 3]
  - name: changed
    pipeline: tensor-input-cpu
    arguments:
      - name: "xs"
        values: [4, -1, 2]
  - name: longer
    pipeline: tensor-input-cpu
    arguments:
      - name: "xs"
        values: [-2147483648, 2147483647, -1, 0, 1, 46341]
  - name: front
    pipeline: tensor-input-ir
    arguments:
      - name: "xs"
        values: [-2, 0, 3]
  - name: front-changed
    pipeline: tensor-input-ir
    arguments:
      - name: "xs"
        values: [4, -1, 2]
  - name: gpu-first
    pipeline: tensor-input-metal
    threads-per-group: 4
    arguments:
      - name: "xs"
        values: [-2, 0, 3]
  - name: gpu-changed
    pipeline: tensor-input-metal
    threads-per-group: 4
    arguments:
      - name: "xs"
        values: [4, -1, 2]
  - name: gpu-longer
    pipeline: tensor-input-metal
    threads-per-group: 4
    arguments:
      - name: "xs"
        values: [-2147483648, 2147483647, -1, 0, 1, 46341]
*)
(* @assert:first:result
[5, 1, 10]
*)
(* @assert:changed:result
[17, 2, 5]
*)
(* @assert:longer:result
[1, 2, 2, 1, 2, -2147479014]
*)
(* @assert:first:ir-check
CHECK: llvm.func @_mlir_ciface_tensor_expression
CHECK-NOT: llvm.call @malloc
CHECK-NOT: llvm.call @free
*)
(* @assert:front:ir-equal
module {
  func.func @tensor_expression(%input: memref<3xi32>, %output: memref<3xi32>) attributes {llvm.emit_c_interface} {
    "letalg.tensor_map"(%input, %output) ({
    ^bb0(%x: i32):
      %square = arith.muli %x, %x : i32
      %one = arith.constant 1 : i32
      %result = arith.addi %square, %one : i32
      letalg.yield %result : i32
    }) : (memref<3xi32>, memref<3xi32>) -> ()
    func.return
  }
}
*)
(* @assert:front-changed:ir-equal
module {
  func.func @tensor_expression(%input: memref<3xi32>, %output: memref<3xi32>) attributes {llvm.emit_c_interface} {
    "letalg.tensor_map"(%input, %output) ({
    ^bb0(%x: i32):
      %square = arith.muli %x, %x : i32
      %one = arith.constant 1 : i32
      %result = arith.addi %square, %one : i32
      letalg.yield %result : i32
    }) : (memref<3xi32>, memref<3xi32>) -> ()
    func.return
  }
}
*)
(* @assert:gpu-first:result
[5, 1, 10]
*)
(* @assert:gpu-first:compare
first
*)
(* @assert:gpu-changed:result
[17, 2, 5]
*)
(* @assert:gpu-changed:compare
changed
*)
(* @assert:gpu-longer:result
[1, 2, 2, 1, 2, -2147479014]
*)
(* @assert:gpu-longer:compare
longer
*)
xs * xs + 1
