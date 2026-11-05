(* @test
version: 1
input: source
runs:
  - name: whitespace
    pipeline: tensor-input-cpu
    arguments:
      - name: "xs"
        values: [-2, 0, 3]
*)
(* @assert:whitespace:result
[5, 1, 10]
*)
xs	*	 xs
 + 1
