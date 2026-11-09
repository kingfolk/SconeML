#!/usr/bin/env python3
"""M3 acceptance: seeded shape/value fixtures and complete hand-authored MSL."""
import argparse
from math import prod
from pathlib import Path
import sys
sys.dont_write_bytecode = True
from generate_broadcast import reference, check_oracle, signed, MASK, MATRIX, ROW, COLUMN, THREE

ROOT = Path(__file__).resolve().parents[1] / 'acceptance_m3'
GOLDENS = Path(__file__).resolve().parent / 'msl'
SEED = 0x5C0E0003
ANCHORS = [-2147483648, 2147483647, -1, 0, 1]


def values(count, stream):
    state = (SEED ^ (stream * 0x9E3779B9)) & MASK
    result = []
    for index in range(count):
        state = (1664525 * state + 1013904223) & MASK
        result.append(ANCHORS[(index + stream) % len(ANCHORS)] if index < 5 else signed(state))
    return result


def argument(name, shape, stream):
    return name, values(prod(shape), stream), shape


def polynomial():
    return 'a * a + 17 * a - 2147483647', [
        ('aa','mul','a','a'), ('n','constant',17), ('na','mul','n','a'),
        ('sum','add','aa','na'), ('max','constant',2147483647), ('out','sub','sum','max')]


def ternary():
    return 'a * a + c - a * b - b', [
        ('aa','mul','a','a'), ('ac','add','aa','c'), ('ab','mul','a','b'),
        ('difference','sub','ac','ab'), ('out','sub','difference','b')]


def case(name, source, args, ops=(), result='out', width=None, golden=None):
    runs = [('cpu','shaped-input-cpu'),('metal','shaped-input-metal')]
    if golden:
        runs += [('device','shaped-input-gpu-ir'),('msl','shaped-input-msl')]
    text = f'(* @case {name} *)\n(* @test\nversion: 1\ninput: source\nruns:\n'
    for run, pipeline in runs:
        text += f'  - name: {run}\n    pipeline: {pipeline}\n'
        if run == 'metal' and width is not None:
            text += f'    threads-per-group: {width}\n'
        text += '    arguments:\n'
        for variable, data, shape in args:
            text += f'      - name: {variable}\n        values: {data}\n        shape: {shape}\n'
    text += '*)\n'
    shape, expected = reference(args, ops, result)
    for run in ('cpu','metal'):
        text += f'(* @assert:{run}:result\n{expected}\n*)\n(* @assert:{run}:shape\n{shape}\n*)\n'
    text += '(* @assert:metal:compare\ncpu\n*)\n'
    if golden:
        index_op = 'arith.remui' if golden == 'broadcast-row-column.msl' else 'arith.constant 0 : index'
        text += f'''(* @assert:device:ir-check
CHECK: module attributes {{sconeml.broadcast =
CHECK: gpu.func @tensor_map
CHECK: %[[I:.*]] = arith.addi
CHECK: %[[N:.*]] = arith.index_castui
CHECK: %[[P:.*]] = arith.cmpi ult, %[[I]], %[[N]]
CHECK: scf.if %[[P]] {{
CHECK: memref.load %arg0[%[[I]]]
CHECK: {index_op}
CHECK: memref.load %arg1
CHECK: memref.store {{{{.*}}}}, %arg{len(args)}[%[[I]]]
*)
(* @assert:device:source-unchanged
true
*)
(* @assert:msl:text-equal
{(GOLDENS/golden).read_text()}*)
(* @assert:msl:source-unchanged
true
*)
'''
    return text + source + '\n'


def matrix_shape(count):
    for rows in range(2, min(count, 16) + 1):
        if count % rows == 0:
            return [rows, count // rows]
    return [1,count]


def fixtures():
    groups = {name: [] for name in ('semantics.ml','grid-boundaries.ml','empty.ml')}
    def add(file, *args, **kwargs):
        groups[file].append(case(*args, **kwargs))
    add('semantics.ml','row-column-golden','a + b + c',[MATRIX,ROW,COLUMN],THREE,
        golden='broadcast-row-column.msl')
    source, ops = polynomial()
    add('semantics.ml','rank1-polynomial',source,[argument('a',[7],0)],ops)
    add('semantics.ml','single-row','b - a',[argument('a',[1,7],1),argument('b',[7],2)], [('out','sub','b','a')])
    add('semantics.ml','single-column','a * b',[argument('a',[7,1],3),argument('b',[1],4)], [('out','mul','a','b')])
    add('semantics.ml','outer-nonsquare','a * b + 17',
        [argument('a',[3,1],5),argument('b',[1,5],6)],
        [('ab','mul','a','b'),('n','constant',17),('out','add','ab','n')])
    add('semantics.ml','unused-shapes','a - 1',
        [argument('b',[3,5],7),argument('a',[3,1],8),argument('c',[5],9)], [('one','constant',1),('out','sub','a','one')])
    add('semantics.ml','constant-min','0 - 2147483647 - 1',
        [argument('c',[5],10),argument('a',[3,1],11),argument('b',[3,5],12)], [('out','constant',-2147483648)])
    add('semantics.ml','singleton-golden','a * b',
        [argument('a',[2,3],13),argument('b',[1,1],11)], [('out','mul','a','b')],
        golden='broadcast-singleton.msl')
    source, ops = ternary()
    add('semantics.ml','reordered-shared-scalar',source,
        [argument('c',[3,1],15),argument('b',[5],16),argument('a',[3,5],17)],ops)
    for width in (4,32,256):
        for position, count in enumerate((1,width-1,width,width+1,2*width+3)):
            shape = matrix_shape(count)
            rows, columns = shape
            stream = width * 10 + position * 3
            if position == 0:
                source, ops = polynomial()
                args = [argument('a',[1],stream)]
            elif position == 1:
                source, ops = 'b - a', [('out','sub','b','a')]
                args = [argument('b',[columns],stream+1),argument('a',shape,stream)]
            elif position == 2:
                source, ops = ternary()
                args = [argument('a',[rows,1],stream),argument('b',[1,columns],stream+1),argument('c',shape,stream+2)]
            elif position == 3:
                source, ops = polynomial()
                args = [argument('a',shape,stream)]
            else:
                source, ops = ternary()
                args = [argument('c',[rows,1],stream+2),argument('a',shape,stream),argument('b',[columns],stream+1)]
            assert prod(reference(args,ops,'out')[0]) == count
            add('grid-boundaries.ml',f'w{width}-n{count:03d}',source,args,ops,width=width)
    empty = [
        ('rank1','a',[('a',[],[0])],[],'a'),
        ('rows','a + b',[('a',[],[0,5]),argument('b',[1,5],18)],[('out','add','a','b')],'out'),
        ('columns','a + b',[argument('a',[3,1],19),('b',[],[1,0])],[('out','add','a','b')],'out'),
        ('both-axes','c',[('b',[],[1,0]),argument('c',[1],20),('a',[],[0,1])],[],'c'),
        ('unused-empty','a',[argument('a',[1],21),('b',[],[0,3]),argument('c',[3],22)],[],'a'),
        ('constant','7',[argument('a',[1],23),('b',[],[0])],[('out','constant',7)],'out')]
    for name, source, args, ops, result in empty:
        add('empty.ml',name,source,args,ops,result)
    return {name:f'(* Generated by test/reference/generate_m3.py; seed 0x{SEED:08x}. *)\n\n'+'\n'.join(cases)
            for name,cases in groups.items()}


def check_acceptance_reference():
    check_oracle()
    if values(5,0) != ANCHORS or values(9,100) == values(9,101):
        raise ValueError('seeded input anchor/stream independence failed')
    shape, result = reference([('a',[2,-1],[2,1]),('b',[3,5],[2])],
        [('aa','mul','a','a'),('ab','mul','a','b'),('out','sub','aa','ab')],'out')
    if (shape,result) != ([2,2],[-2,-6,4,6]):
        raise ValueError('M3 indexing/scalar anchor failed')
    source, ops = polynomial()
    if reference([('a',[2147483647,-2147483648],[2])],ops,'out') != ([2],[-15,1]):
        raise ValueError('M3 wrapping polynomial anchor failed')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    check_acceptance_reference()
    expected=fixtures()
    if args.check:
        actual={str(path.relative_to(ROOT)) for path in ROOT.rglob('*') if path.is_file()}
        if actual != set(expected):
            raise SystemExit(f'M3 acceptance fixture set differs: {actual ^ set(expected)}')
        for name,text in expected.items():
            if (ROOT/name).read_text()!=text:
                raise SystemExit(f'stale M3 acceptance fixture: {name}')
        print(f'PASS M3 oracle anchors, seed 0x{SEED:08x}, 3 fixture files and 2 complete MSL expectations')
    else:
        ROOT.mkdir(parents=True,exist_ok=True)
        for name,text in expected.items():
            (ROOT/name).write_text(text)
        print('Generated 3 M3 acceptance fixture files')

if __name__=='__main__':
    main()
