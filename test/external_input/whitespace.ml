(* @test
version: 1
input: source
runs:
  - name: whitespace
    pipeline: tensor-input-cpu
    arguments:
      - name: "xs"
        values: [-2, 0, 3]
  - name: gpu-whitespace
    pipeline: tensor-input-metal
    threads-per-group: 4
    arguments:
      - name: "xs"
        values: [-2, 0, 3]
*)
(* @assert:whitespace:result
[5, 1, 10]
*)
(* @assert:gpu-whitespace:result
[5, 1, 10]
*)
(* @assert:gpu-whitespace:compare
whitespace
*)
xs	*	 xs
 + 1
