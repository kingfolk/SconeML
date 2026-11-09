#!/usr/bin/env python3
"""Independent nested-loop oracle and reproducible M3-B source fixtures.

No production parser, shape function, compiler output or backend is used. Source
expressions are paired with separately specified scalar instruction lists.
--check is read-only and verifies explicit oracle anchors and exact fixture text.
"""
import argparse
from itertools import permutations
from math import prod
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'shaped_input' / 'broadcast'
MASK = (1 << 32) - 1


def signed(value):
    value &= MASK
    return value if value < (1 << 31) else value - (1 << 32)


def reference(arguments, instructions, result):
    for _, values, shape in arguments:
        if len(shape) not in (1, 2) or any(n < 0 for n in shape):
            raise ValueError('invalid shape')
        if len(values) != prod(shape):
            raise ValueError('invalid storage')
    rank = max(len(shape) for _, _, shape in arguments)
    padded = [(1,) * (rank - len(shape)) + tuple(shape) for _, _, shape in arguments]
    shape = []
    # Independent set-based inference: zero is a non-singleton extent.
    for extents in zip(*padded):
        distinct = set(extents) - {1}
        if len(distinct) > 1:
            raise ValueError('incompatible broadcast')
        shape.append(distinct.pop() if distinct else 1)
    rows, columns = (1, shape[0]) if rank == 1 else shape
    output = []
    for row in range(rows):
        for column in range(columns):
            env = {}
            for name, values, dims in arguments:
                if len(dims) == 1:
                    index = 0 if dims[0] == 1 else column
                else:
                    r = 0 if dims[0] == 1 else row
                    c = 0 if dims[1] == 1 else column
                    index = r * dims[1] + c
                env[name] = values[index] & MASK
            for name, op, *operands in instructions:
                if op == 'constant':
                    value = operands[0]
                elif op == 'add':
                    value = env[operands[0]] + env[operands[1]]
                elif op == 'sub':
                    value = env[operands[0]] - env[operands[1]]
                elif op == 'mul':
                    value = env[operands[0]] * env[operands[1]]
                else:
                    raise ValueError(op)
                env[name] = value & MASK
            output.append(signed(env[result]))
    return shape, output


ADD = [('out', 'add', 'a', 'b')]
THREE = [('ab', 'add', 'a', 'b'), ('out', 'add', 'ab', 'c')]
NONCOMMUTATIVE = [('ab', 'mul', 'a', 'b'), ('ac', 'add', 'ab', 'c'),
                  ('out', 'sub', 'ac', 'a')]
MATRIX = ('a', [1, 2, 3, 4, 5, 6], [2, 3])
ROW = ('b', [10, 20, 30], [3])
COLUMN = ('c', [100, 200], [2, 1])


def check_oracle():
    anchors = [
        ([MATRIX, ROW, COLUMN], THREE, 'out', [2, 3], [111, 122, 133, 214, 225, 236]),
        ([('a', [2, 3], [2, 1]), ('b', [10, 20, 30], [1, 3])], ADD, 'out',
         [2, 3], [12, 22, 32, 13, 23, 33]),
        ([('a', [2147483647, -2147483648], [2, 1]), ('b', [2, -1], [2])],
         [('out', 'mul', 'a', 'b')], 'out', [2, 2], [-2, -2147483647, 0, -2147483648]),
        ([('a', [], [1, 0]), ('b', [1, 2], [2, 1])], ADD, 'out', [2, 0], []),
        ([('a', [9], [1]), ('b', [], [0])], ADD, 'out', [0], []),
    ]
    for arguments, ops, result, shape, values in anchors:
        if reference(arguments, ops, result) != (shape, values):
            raise ValueError('broadcast oracle anchor failed')
    for a, b in [([0, 3], [2, 3]), ([2, 3], [2]), ([0, 2], [0, 3])]:
        try:
            reference([('a', [0] * prod(a), a), ('b', [0] * prod(b), b)], ADD, 'out')
        except ValueError:
            pass
        else:
            raise ValueError('oracle accepted incompatible shapes')


def case(name, source, arguments, instructions=None, result='out', error=None, check=None):
    text = f'(* @case {name} *)\n(* @test\nversion: 1\ninput: source\nruns:\n'
    pipelines = [('cpu', 'shaped-input-cpu')]
    if check:
        pipelines.append(('ir', 'shaped-input-ir'))
    for run, pipeline in pipelines:
        text += f'  - name: {run}\n    pipeline: {pipeline}\n    arguments:\n'
        for variable, values, shape in arguments:
            text += f'      - name: {variable}\n        values: {values}\n        shape: {shape}\n'
    text += '*)\n'
    if error:
        text += f'(* @assert:cpu:error\nstage: frontend\nmessage: {error}\n*)\n'
    else:
        shape, output = reference(arguments, instructions, result)
        text += f'(* @assert:cpu:result\n{output}\n*)\n(* @assert:cpu:shape\n{shape}\n*)\n'
    if check:
        text += f'(* @assert:ir:ir-check\n{check}\n*)\n'
    return text + source + '\n'


THREE_CHECK = '''CHECK-DAG: #[[M:[a-zA-Z0-9_]+]] = affine_map<(d0, d1) -> (d0, d1)>
CHECK-DAG: #[[R:[a-zA-Z0-9_]+]] = affine_map<(d0, d1) -> (d1)>
CHECK-DAG: #[[C:[a-zA-Z0-9_]+]] = affine_map<(d0, d1) -> (d0, 0)>
CHECK: func.func @shaped_tensor_expression
CHECK-SAME: memref<6xi32>
CHECK-SAME: memref<3xi32>
CHECK-SAME: memref<2xi32>
CHECK-SAME: memref<6xi32>
CHECK: %[[A:.*]] = memref.reinterpret_cast %arg0
CHECK-SAME: sizes: [2, 3], strides: [3, 1]
CHECK: %[[B:.*]] = memref.reinterpret_cast %arg1
CHECK-SAME: sizes: [3], strides: [1]
CHECK: %[[CIN:.*]] = memref.reinterpret_cast %arg2
CHECK-SAME: sizes: [2, 1], strides: [1, 1]
CHECK: %[[OUT:.*]] = memref.reinterpret_cast %arg3
CHECK: linalg.generic {indexing_maps = [#[[M]], #[[R]], #[[C]], #[[M]]], iterator_types = ["parallel", "parallel"]}
CHECK-SAME: ins(%[[A]], %[[B]], %[[CIN]]
CHECK-SAME: outs(%[[OUT]]
CHECK: ^bb0(%[[AV:.*]]: i32, %[[BV:.*]]: i32, %[[CV:.*]]: i32, %[[OV:.*]]: i32):
CHECK-NEXT: %[[AB:.*]] = arith.addi %[[AV]], %[[BV]] : i32
CHECK-NEXT: %[[ABC:.*]] = arith.addi %[[AB]], %[[CV]] : i32
CHECK-NEXT: linalg.yield %[[ABC]] : i32
CHECK-NEXT: }
CHECK-NEXT: return'''


def fixtures():
    groups = {'semantics.ml': [], 'empty.ml': [], 'invalid.ml': []}
    def add(file, *args, **kwargs):
        groups[file].append(case(*args, **kwargs))
    add('semantics.ml', 'matrix-row-column', 'a + b + c', [MATRIX, ROW, COLUMN], THREE,
        check=THREE_CHECK)
    for index, arguments in enumerate(permutations([MATRIX, ROW, COLUMN])):
        add('semantics.ml', f'argument-order-{index}', 'a * b + c - a', arguments, NONCOMMUTATIVE)
    for name, a, b in [
        ('row-vector', MATRIX, ROW),
        ('column', MATRIX, ('b', [10, 20], [2, 1])),
        ('outer', ('a', [1, 2], [2, 1]), ('b', [10, 20, 30], [1, 3])),
        ('vector-column-outer', ('a', [1, 2, 3], [3]), ('b', [10, 20], [2, 1])),
        ('singleton-rank1', ('a', [1, 2, 3], [3]), ('b', [10], [1])),
        ('singleton-first', ('a', [10], [1]), ('b', [1, 2, 3], [3])),
        ('singleton-rank2', MATRIX, ('b', [10], [1, 1])),
        ('row-rank2', MATRIX, ('b', [10, 20, 30], [1, 3])),
        ('output-rank-from-unused', ('a', [10], [1]), ('b', [1, 2, 3], [1, 3])),
    ]:
        if name == 'output-rank-from-unused':
            add('semantics.ml', name, 'a', [a, b], [], result='a')
        else:
            add('semantics.ml', name, 'a + b', [a, b], ADD)
    add('semantics.ml', 'constant-outer', '7 * 3 - 1',
        [('a', [1, 2], [2, 1]), ('b', [3, 4, 5], [1, 3])], [('out', 'constant', 20)])
    add('semantics.ml', 'unused-enlarges-output', 'a + a', [('a', ROW[1], ROW[2]),
        ('b', [1, 2], [2, 1])], [('out', 'add', 'a', 'a')])
    add('semantics.ml', 'wrapping-outer', 'a * b',
        [('a', [2147483647, -2147483648], [2, 1]), ('b', [2, -1], [2])], [('out', 'mul', 'a', 'b')])
    add('semantics.ml', 'overlapping-broadcast', 'a * b + c',
        [MATRIX, ('b', [2, 3, 4], [3]), ('c', [10, 20], [2, 1])],
        [('ab', 'mul', 'a', 'b'), ('out', 'add', 'ab', 'c')])
    # Displaced signed data and nonsquare loops exercise indexing beyond tiny examples.
    for rows, columns in [(1, 7), (7, 1), (3, 5), (5, 3), (2, 17)]:
        args = [('a', [signed(i * 0x9E3779B9) for i in range(rows * columns)], [rows, columns]),
                ('b', [signed(17 - i * 65537) for i in range(columns)], [columns]),
                ('c', [signed(i * 2147483647) for i in range(rows)], [rows, 1])]
        add('semantics.ml', f'nonsquare-{rows}x{columns}', 'a * b + c - a', args, NONCOMMUTATIVE)
    for name, a, b in [
        ('zero-rows', ('a', [], [0, 3]), ('b', [1, 2, 3], [1, 3])),
        ('zero-columns', ('a', [], [1, 0]), ('b', [1, 2], [2, 1])),
        ('nonempty-first', ('a', [1, 2], [2, 1]), ('b', [], [1, 0])),
        ('zero-vector', ('a', [], [0]), ('b', [9], [1])),
        ('singleton-first-zero', ('a', [9], [1]), ('b', [], [0])),
        ('zero-both-axes', ('a', [], [0, 1]), ('b', [], [1, 0])),
        ('zero-trailing-missing-axis', ('a', [], [0]), ('b', [1, 2], [2, 1])),
    ]:
        add('empty.ml', name, 'a + b', [a, b], ADD)
    add('empty.ml', 'constant-empty', '7', [('a', [1, 2], [2, 1]), ('b', [], [0])], [('out', 'constant', 7)])
    for name, args in [
        ('zero-vs-two', [('a', [], [0, 3]), ('b', [1, 2, 3, 4, 5, 6], [2, 3])]),
        ('rank1-is-not-column', [MATRIX, ('b', [1, 2], [2])]),
        ('unused-incompatible', [MATRIX, ('b', [1, 2], [2])]),
        ('zero-trailing-incompatible', [('a', [], [2, 0]), ('b', [1, 2, 3], [3])]),
        ('empty-third-incompatible', [('a', [], [0, 3]), ('b', [1, 2, 3], [3]), ('c', [1, 2], [2, 1])]),
    ]:
        add('invalid.ml', name, 'a', args, error='incompatible shaped tensor broadcast dimensions')
    add('invalid.ml', 'nonempty-unused-storage', '7', [('a', [], [0, 3]), ('b', [1], [1, 3])],
        error='storage count does not match')
    add('invalid.ml', 'empty-still-validates-syntax', 'a +',
        [('a', [1, 2], [2, 1]), ('b', [], [1, 0])], error='empty or missing an operand')
    add('invalid.ml', 'empty-still-validates-names', 'unknown',
        [('a', [1, 2], [2, 1]), ('b', [], [1, 0])], error='unknown tensor expression variable')
    return {file: '(* Generated by test/reference/generate_broadcast.py; do not edit. *)\n\n' + '\n'.join(cases)
            for file, cases in groups.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    check_oracle()
    expected = fixtures()
    if args.check:
        actual = {str(path.relative_to(ROOT)) for path in ROOT.rglob('*') if path.is_file()}
        if actual != set(expected):
            raise SystemExit(f'broadcast fixture set differs: {actual ^ set(expected)}')
        for name, content in expected.items():
            if (ROOT / name).read_text() != content:
                raise SystemExit(f'stale broadcast fixture: {name}')
        print(f'PASS broadcast independent oracle anchors and {len(expected)} generated fixtures')
    else:
        ROOT.mkdir(parents=True, exist_ok=True)
        for name, content in expected.items():
            (ROOT / name).write_text(content)
        print(f'Generated {len(expected)} broadcast fixture files')


if __name__ == '__main__':
    main()
