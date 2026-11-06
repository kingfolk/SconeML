(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: tensor-input-cpu
    arguments:
      - name: xs
        values: [-2147483648]
  - name: metal
    pipeline: tensor-input-metal
    threads-per-group: 1
    arguments:
      - name: xs
        values: [-2147483648]
  - name: metal-auto
    pipeline: tensor-input-metal
    arguments:
      - name: xs
        values: [-2147483648]
*)
(* @assert:cpu:result
[1]
*)
(* @assert:metal:result
[1]
*)
(* @assert:metal:compare
cpu
*)
(* @assert:metal-auto:result
[1]
*)
(* @assert:metal-auto:compare
cpu
*)
(* @assert:metal:text-check
CHECK: threadgroup_position_in_grid
CHECK: threads_per_threadgroup
CHECK: thread_position_in_threadgroup
CHECK: const uint v0 = group_id.x;
*)
xs * xs + 17 * xs - 2147483647
