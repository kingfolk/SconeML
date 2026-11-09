#!/usr/bin/env python3
"""Independent nested-loop broadcast CPU/Metal fixtures; --check never writes."""
import argparse
from itertools import permutations
from pathlib import Path
import sys
sys.dont_write_bytecode = True
from generate_broadcast import reference, check_oracle, signed, MATRIX, ROW, COLUMN, THREE, NONCOMMUTATIVE

ROOT = Path(__file__).resolve().parents[1] / 'broadcast_gpu'


def case(name, source, arguments, instructions=(), result='out', error=None, width=None, structural=False):
    runs = [('cpu', 'shaped-input-cpu'), ('metal', 'shaped-input-metal')]
    if error:
        runs = [('metal', 'shaped-input-metal')]
    if structural:
        runs += [('device', 'shaped-input-gpu-ir'), ('msl', 'shaped-input-msl')]
    text = f'(* @case {name} *)\n(* @test\nversion: 1\ninput: source\nruns:\n'
    for run, pipeline in runs:
        text += f'  - name: {run}\n    pipeline: {pipeline}\n'
        if run == 'metal' and width is not None:
            text += f'    threads-per-group: {width}\n'
        text += '    arguments:\n'
        for variable, values, shape in arguments:
            text += f'      - name: {variable}\n        values: {values}\n        shape: {shape}\n'
    text += '*)\n'
    if error:
        text += f'(* @assert:metal:error\nstage: {error[0]}\nmessage: {error[1]}\n*)\n'
    else:
        shape, output = reference(arguments, instructions, result)
        for run in ('cpu', 'metal'):
            text += f'(* @assert:{run}:result\n{output}\n*)\n(* @assert:{run}:shape\n{shape}\n*)\n'
        text += '(* @assert:metal:compare\ncpu\n*)\n'
    if structural:
        text += '''(* @assert:device:ir-check
CHECK: module attributes {sconeml.broadcast = {input_counts = array<i64: 6, 3, 2>, input_shapes = [array<i64: 2, 3>, array<i64: 3>, array<i64: 2, 1>], output_count = 6 : i64, shape = array<i64: 2, 3>}}
CHECK: gpu.func @tensor_map
CHECK-SAME: sconeml.broadcast_kernel
CHECK: %[[I:.*]] = arith.addi
CHECK: %[[N:.*]] = arith.index_castui
CHECK: %[[BOUND:.*]] = arith.cmpi ult, %[[I]], %[[N]]
CHECK: scf.if %[[BOUND]] {
CHECK-NEXT: %[[A:.*]] = memref.load %arg0[%[[I]]]
CHECK-NEXT: %[[C:.*]] = arith.constant 3 : index
CHECK-NEXT: %[[COL:.*]] = arith.remui %[[I]], %[[C]] : index
CHECK-NEXT: %[[B:.*]] = memref.load %arg1[%[[COL]]]
CHECK-NEXT: %[[C2:.*]] = arith.constant 3 : index
CHECK-NEXT: %[[ROW:.*]] = arith.divui %[[I]], %[[C2]] : index
CHECK-NEXT: %[[CV:.*]] = memref.load %arg2[%[[ROW]]]
CHECK-NEXT: %[[AB:.*]] = arith.addi %[[A]], %[[B]] : i32
CHECK-NEXT: %[[ABC:.*]] = arith.addi %[[AB]], %[[CV]] : i32
CHECK-NEXT: memref.store %[[ABC]], %arg3[%[[I]]]
CHECK-NEXT: }
*)
(* @assert:device:source-unchanged
true
*)
(* @assert:msl:source-unchanged
true
*)
(* @assert:msl:text-check
CHECK: device const uint *input0
CHECK-SAME: buffer(0)
CHECK: device const uint *input1
CHECK-SAME: buffer(1)
CHECK: device const uint *input2
CHECK-SAME: buffer(2)
CHECK: device uint *output
CHECK-SAME: buffer(3)
CHECK: constant uint &count
CHECK-SAME: buffer(4)
CHECK: if (
CHECK: input0[
CHECK: = 3u;
CHECK: %
CHECK: input1[
CHECK: = 3u;
CHECK: /
CHECK: input2[
CHECK: output[
*)
'''
    return text + source + '\n'


def fixtures():
    groups = {name: [] for name in ('semantics.ml', 'empty.ml', 'invalid.ml', 'launch-boundaries.ml')}
    def add(file, *args, **kwargs):
        groups[file].append(case(*args, **kwargs))
    add('semantics.ml', 'matrix-row-column', 'a + b + c', [MATRIX, ROW, COLUMN], THREE, structural=True)
    for i, args in enumerate(permutations([MATRIX, ROW, COLUMN])):
        add('semantics.ml', f'binding-order-{i}', 'a * b + c - a', args, NONCOMMUTATIVE)
    addops = [('out', 'add', 'a', 'b')]
    for name, a, b in [
        ('row', MATRIX, ROW), ('column', MATRIX, ('b', [10,20], [2,1])),
        ('outer', ('a',[1,2],[2,1]), ('b',[10,20,30],[1,3])),
        ('vector-column', ('a',[1,2,3],[3]), ('b',[10,20],[2,1])),
        ('singleton', ('a',[1,2,3],[3]), ('b',[10],[1])),
        ('singleton-first', ('a',[10],[1]), ('b',[1,2,3],[3])),
        ('singleton-rank2', MATRIX, ('b',[7],[1,1])),
        ('rank-change-identity', ('a',[1,2,3],[3]), ('b',[10,20,30],[1,3])),
        ('single-column-identity', ('a',[1,2,3],[3,1]), ('b',[10],[1])),
        ('single-row-identity', ('a',[1,2,3],[1,3]), ('b',[10],[1]))]:
        add('semantics.ml', name, 'a + b', [a,b], addops)
    outer = [('a',[1,2],[2,1]), ('b',[10,20,30],[1,3])]
    add('semantics.ml','constant','7 * 3 - 1',outer,[('out','constant',20)])
    add('semantics.ml','unused-enlarges','a + a',outer,[('out','add','a','a')])
    add('semantics.ml','wrapping','a * b',[('a',[2147483647,-2147483648],[2,1]),('b',[2,-1],[2])],[('out','mul','a','b')])
    add('semantics.ml','overlap','a * b + c',
        [MATRIX,('b',[2,3,4],[3]),('c',[1,2],[2,1])],[('ab','mul','a','b'),('out','add','ab','c')])
    for name, args in [
        ('zero-rows',[('a',[],[0,3]),('b',[1,2,3],[1,3])]),
        ('zero-columns',[('a',[],[1,0]),('b',[1,2],[2,1])]),
        ('nonempty-first',[('a',[1,2],[2,1]),('b',[],[1,0])]),
        ('zero-vector',[('a',[],[0]),('b',[7],[1])]),
        ('singleton-first',[('a',[7],[1]),('b',[],[0])]),
        ('both-axes-zero',[('a',[],[0,1]),('b',[],[1,0])]),
        ('missing-leading-axis',[('a',[],[0]),('b',[1,2],[2,1])]),
        ('constant',[('a',[1,2],[2,1]),('b',[],[0])])]:
        add('empty.ml', name, '7' if name=='constant' else 'a + b', args,
            [('out','constant',7)] if name=='constant' else addops)
    for width in (1,4,32):
        for rows, columns in ((2,3),(3,5),(5,3),(2,17),(1,33)):
            args=[('a',[signed(i*0x9e3779b9) for i in range(rows*columns)],[rows,columns]),
                  ('b',[signed(7-i*65537) for i in range(columns)],[columns]),
                  ('c',[signed(i*2147483647) for i in range(rows)],[rows,1])]
            add('launch-boundaries.ml',f'w{width}-{rows}x{columns}','a * b + c - a',args,NONCOMMUTATIVE,width=width)
    for name, args in [
        ('orientation',[MATRIX,('b',[1,2],[2])]),
        ('zero-vs-two',[('a',[],[0,3]),('b',[1,2,3,4,5,6],[2,3])]),
        ('unused-incompatible',[MATRIX,('b',[1,2],[2])])]:
        add('invalid.ml',name,'a',args,error=('frontend','incompatible shaped tensor broadcast dimensions'))
    add('invalid.ml','empty-storage','7',[('a',[],[0,3]),('b',[1],[3])],error=('frontend','storage count does not match'))
    add('invalid.ml','empty-syntax','a +',[('a',[1,2],[2,1]),('b',[],[0])],error=('frontend','empty or missing an operand'))
    add('invalid.ml','empty-invalid-width','a + b',[('a',[1,2],[2,1]),('b',[],[0])],width=4294967296,
        error=('gpu-lowering','threads per threadgroup'))
    return {file: '(* Generated by test/reference/generate_gpu_broadcast.py; do not edit. *)\n\n'+'\n'.join(cases)
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
            raise SystemExit(f'GPU broadcast fixture set differs: {actual ^ set(expected)}')
        for name, text in expected.items():
            if (ROOT/name).read_text() != text:
                raise SystemExit(f'stale GPU broadcast fixture: {name}')
        print('PASS GPU broadcast independent oracle and 4 generated fixtures')
    else:
        ROOT.mkdir(parents=True, exist_ok=True)
        for name, text in expected.items():
            (ROOT/name).write_text(text)
        print('Generated 4 GPU broadcast fixtures')

if __name__ == '__main__':
    main()
