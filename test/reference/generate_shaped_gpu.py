#!/usr/bin/env python3
"""M3-C shaped Metal fixtures, using the independent nested-loop i32 oracle."""
import argparse
from pathlib import Path
import sys
sys.dont_write_bytecode = True
from generate_broadcast import reference, check_oracle, signed

ROOT = Path(__file__).resolve().parents[1] / 'shaped_gpu'


def case(name, source, args, ops=(), result='out', error=None, width=None, structural=False):
    runs = [('cpu', 'shaped-input-cpu'), ('metal', 'shaped-input-metal')]
    if error: runs = [('metal', 'shaped-input-metal')]
    if structural: runs += [('device', 'shaped-input-gpu-ir'), ('msl', 'shaped-input-msl')]
    text = f'(* @case {name} *)\n(* @test\nversion: 1\ninput: source\nruns:\n'
    for run, pipeline in runs:
        text += f'  - name: {run}\n    pipeline: {pipeline}\n'
        if run == 'metal' and width is not None: text += f'    threads-per-group: {width}\n'
        text += '    arguments:\n'
        for var, values, shape in args:
            text += f'      - name: {var}\n        values: {values}\n        shape: {shape}\n'
    text += '*)\n'
    if error:
        stage, message = error
        text += f'(* @assert:metal:error\nstage: {stage}\nmessage: {message}\n*)\n'
    else:
        shape, output = reference(args, ops, result)
        for run in ('cpu', 'metal'):
            text += f'(* @assert:{run}:result\n{output}\n*)\n(* @assert:{run}:shape\n{shape}\n*)\n'
        text += '(* @assert:metal:compare\ncpu\n*)\n'
    if structural:
        k = len(args)
        count = len(args[0][1])
        text += f'''(* @assert:device:ir-check
CHECK: module attributes {{sconeml.shaped = {{input_counts = array<i64: {', '.join([str(count)] * k)}>, output_count = {count} : i64, shape = array<i64: {', '.join(map(str,args[0][2]))}>}}}}
CHECK: gpu.module @kernels
CHECK: gpu.func @tensor_map
CHECK: gpu.block_id x
CHECK: gpu.block_dim x
CHECK: gpu.thread_id x
CHECK: arith.index_castui
CHECK: arith.cmpi ult
CHECK: scf.if
CHECK: memref.load %arg0
CHECK: memref.load %arg1
CHECK: memref.load %arg2
CHECK: arith.muli
CHECK: arith.addi
CHECK: memref.store {{{{.*}}}}, %arg3
CHECK: gpu.return
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
CHECK: input1[
CHECK: input2[
CHECK: output[
*)
'''
    return text + source + '\n'


def fixtures():
    groups = {'semantics.ml': [], 'launch-boundaries.ml': [], 'invalid.ml': []}
    def add(file, *args, **kw): groups[file].append(case(*args, **kw))
    three = [('ab', 'mul', 'a', 'b'), ('ac', 'add', 'ab', 'c'), ('out', 'sub', 'ac', 'a')]
    for rank, shape in [('rank1', [6]), ('rank2', [2, 3])]:
        add('semantics.ml', f'{rank}-three', 'a * b + c - a',
            [('a', [1,2,3,4,5,6], shape), ('b', [10,20,30,40,50,60], shape),
             ('c', [100,200,300,400,500,600], shape)], three, structural=rank=='rank2')
        add('semantics.ml', f'{rank}-one', 'a * a + 1', [('a',[1,2,3,4,5,6],shape)],
            [('aa','mul','a','a'),('one','constant',1),('out','add','aa','one')])
        add('semantics.ml', f'{rank}-two', 'b - a', [('a',[1,2,3,4,5,6],shape),('b',[9,8,7,6,5,4],shape)],
            [('out','sub','b','a')])
    for name, shape, values in [('singleton',[1,1],[7]), ('single-row',[1,3],[1,2,3]), ('single-column',[3,1],[1,2,3])]:
        add('semantics.ml', name, 'a + a', [('a',values,shape)], [('out','add','a','a')])
    add('semantics.ml','wrapping','a * b + 1',
        [('a',[2147483647,-2147483648,-1,65536],[2,2]),('b',[2,-1,-2147483648,65536],[2,2])],
        [('ab','mul','a','b'),('one','constant',1),('out','add','ab','one')])
    for name, source, ops, result in [('constant','7',[('out','constant',7)],'out'),('unused','b',[],'b')]:
        add('semantics.ml',name,source,[('c',[9,8,7,6],[2,2]),('a',[1,2,3,4],[2,2]),('b',[5,6,7,8],[2,2])],ops,result)
    add('semantics.ml','overlap','a * b + c',
        [('a',[1,2,3,4,5,6],[2,3]),('b',[3,4,5,6,7,8],[2,3]),('c',[1,2,3,4,5,6],[2,3])],
        [('ab','mul','a','b'),('out','add','ab','c')])
    for shape in ([0], [0,3], [2,0]):
        add('semantics.ml', 'empty-'+'x'.join(map(str,shape)), 'a + b', [('a',[],shape),('b',[],shape)],
            [('out','add','a','b')])
    for width in (1,4,32):
        for columns in sorted({1,max(1,width-1),width,width+1}):
            # Two rows exercise rank-2 flattening and a padded final group.
            shape=[2,columns];n=2*columns
            args=[('a',[signed(i*0x9e3779b9) for i in range(n)],shape),
                  ('b',[signed(7-i*65537) for i in range(n)],shape)]
            add('launch-boundaries.ml',f'w{width}-n{n}', 'a * b', args,[('out','mul','a','b')],width=width)
    for name, shapes in [('row',([2,3],[3])),('column',([2,3],[2,1])),('outer',([2,1],[1,3])),
                         ('rank-mismatch',([3],[1,3])),('empty-broadcast',([0,3],[1,3]))]:
        from math import prod
        args=[(var,list(range(prod(shape))),shape) for var,shape in zip('ab',shapes)]
        add('invalid.ml',name,'a + b',args,error=('gpu-lowering','shaped GPU broadcasting is not supported yet'))
    add('invalid.ml','empty-bad-syntax','a +',[('a',[],[0,3])],error=('frontend','empty or missing an operand'))
    add('invalid.ml','empty-negative-shape','a',[('a',[],[0,-1])],error=('frontend','extents must be nonnegative'))
    add('invalid.ml','invalid-width','a',[('a',[1],[1])],width=4294967296,error=('gpu-lowering','threads per threadgroup'))
    add('invalid.ml','empty-invalid-width','a',[('a',[],[0])],width=4294967296,error=('gpu-lowering','threads per threadgroup'))
    return {name:'(* Generated by test/reference/generate_shaped_gpu.py; do not edit. *)\n\n'+'\n'.join(cases)
            for name,cases in groups.items()}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--check',action='store_true');args=parser.parse_args()
    check_oracle();expected=fixtures()
    if args.check:
        actual={str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if p.is_file()}
        if actual != set(expected): raise SystemExit(f'shaped GPU fixture set differs: {actual ^ set(expected)}')
        for name,text in expected.items():
            if (ROOT/name).read_text()!=text: raise SystemExit(f'stale shaped GPU fixture: {name}')
        print('PASS shaped GPU independent oracle and 3 generated fixtures')
    else:
        ROOT.mkdir(parents=True,exist_ok=True)
        for name,text in expected.items(): (ROOT/name).write_text(text)
        print('Generated 3 shaped GPU fixtures')

if __name__=='__main__': main()
