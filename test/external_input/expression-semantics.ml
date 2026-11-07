(* @case subtraction *)
(* @test
version: 1
input: source
runs:
  - name: subtract
    pipeline: tensor-input-cpu
    arguments:
      - name: "xs"
        values: [-2147483648, 2147483647, -1, 0, 1, 46341]
  - name: gpu-subtract
    pipeline: tensor-input-metal
    threads-per-group: 4
    arguments:
      - name: "xs"
        values: [-2147483648, 2147483647, -1, 0, 1, 46341]
*)
(* @assert:subtract:result
[2147483647, -2147483648, 0, -1, -2, -46342]
*)
(* @assert:gpu-subtract:result
[2147483647, -2147483648, 0, -1, -2, -46342]
*)
(* @assert:gpu-subtract:compare
subtract
*)
0 - xs - 1

(* @case constant *)
(* @test
version: 1
input: source
runs:
  - name: constant
    pipeline: tensor-input-cpu
    arguments:
      - name: "unused"
        values: [-9, 0, 2147483647]
  - name: empty
    pipeline: tensor-input-cpu
    arguments:
      - name: "unused"
        values: []
  - name: gpu-constant
    pipeline: tensor-input-metal
    threads-per-group: 4
    arguments:
      - name: "unused"
        values: [-9, 0, 2147483647]
  - name: gpu-empty
    pipeline: tensor-input-metal
    threads-per-group: 4
    arguments:
      - name: "unused"
        values: []
*)
(* @assert:constant:result
[42, 42, 42]
*)
(* @assert:empty:result
[]
*)
(* @assert:gpu-constant:result
[42, 42, 42]
*)
(* @assert:gpu-constant:compare
constant
*)
(* @assert:gpu-empty:result
[]
*)
(* @assert:gpu-empty:compare
empty
*)
42

(* @case precedence *)
(* @test
version: 1
input: source
runs:
  - name: precedence
    pipeline: tensor-input-cpu
    arguments:
      - name: "xs"
        values: [-2147483648, 2147483647, -1, 0, 1, 46341]
  - name: gpu-precedence
    pipeline: tensor-input-metal
    threads-per-group: 4
    arguments:
      - name: "xs"
        values: [-2147483648, 2147483647, -1, 0, 1, 46341]
*)
(* @assert:precedence:result
[0, -2, -2, 0, 2, 92682]
*)
(* @assert:gpu-precedence:result
[0, -2, -2, 0, 2, 92682]
*)
(* @assert:gpu-precedence:compare
precedence
*)
xs + 3 * xs - 2 * xs

(* @case whitespace *)
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

(* @case unused-second *)
(* @test
version: 1
input: source
runs:
  - name: unused-second
    pipeline: tensor-input-cpu
    arguments:
      - name: "xs"
        values: [1]
      - name: "ys"
        values: [2]
  - name: gpu-unused-second
    pipeline: tensor-input-metal
    arguments:
      - name: "xs"
        values: [1]
      - name: "ys"
        values: [2]
*)
(* @assert:unused-second:result
[1]
*)
(* @assert:gpu-unused-second:result
[1]
*)(* @assert:gpu-unused-second:compare
unused-second
*)
xs
