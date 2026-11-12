(* @test
version: 1
input: source
runs:
  - name: ir
    pipeline: sum-i32-ir
    arguments:
      - name: a
        values: [1,2,3,4,5,6]
        shape: [2,3]
*)
(* @assert:ir:ir-check
CHECK: sconeml.sum_tree_width = 256
CHECK: linalg.fill
CHECK: iterator_types = ["parallel", "parallel", "reduction"]
CHECK: arith.addi
*)
sum(a, axis=1)
