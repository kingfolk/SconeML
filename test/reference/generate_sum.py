#!/usr/bin/env python3
"""Independent fixed-tree sum fixtures. --check never writes fixture files."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
from itertools import product
import argparse
import runpy
from math import prod

ROOT=Path(__file__).resolve().parents[1]
M4=runpy.run_path(str(Path(__file__).with_name('generate_m4.py')))
MASK=(1<<32)-1

def require(ok,message):
    if not ok: raise RuntimeError(message)

def signed(x):
    x &= MASK
    return x-(1<<32) if x&(1<<31) else x

def canonical(x):
    return 0 if (x&0x7fffffff)<0x00800000 else x

def tree(values,f32):
    values=[canonical(v) if f32 else v&MASK for v in values]
    if not values:return 0
    while True:
        partial=[]
        for start in range(0,len(values),256):
            block=values[start:start+256]
            block += [0]*(256-len(block))
            while len(block)>1:
                block=[canonical(M4['arithmetic']('add',block[i],block[i+1])) if f32 else (block[i]+block[i+1])&MASK for i in range(0,len(block),2)]
            partial.append(block[0])
        if len(partial)==1:return partial[0]
        values=partial

def sums(shape,values,axis,f32):
    if axis is None:return [],[tree(values,f32)]
    out_shape=shape[:axis]+shape[axis+1:]
    outputs=[]
    for coord in product(*(range(d) for d in out_shape)):
        lane=[]
        for k in range(shape[axis]):
            full=list(coord);full.insert(axis,k)
            offset=0
            for d,c in zip(shape,full):offset=offset*d+c
            lane.append(values[offset])
        outputs.append(tree(lane,f32))
    return out_shape,outputs

def array(values,f32):
    return '['+', '.join(f'0x{v:08x}' if f32 else str(signed(v)) for v in values)+']'

def case(name,shape,values,axis=None,f32=False,pipelines=None):
    source='sum(a'+(f', axis={axis}' if axis is not None else '')+')'
    output,result=sums(shape,values,axis,f32)
    text=f'(* @case {name} *)\n(* @test\nversion: 1\ninput: source\nruns:\n'
    pipelines=pipelines or [('cpu',f'sum-{"f32" if f32 else "i32"}-cpu')]
    for run,pipeline in pipelines:
        text+=f'  - name: {run}\n    pipeline: {pipeline}\n'
        if f32:text+='    profile: f32-sum-tree256-ftz\n'
        text+=f'    arguments:\n      - name: a\n        dtype: {"f32" if f32 else "i32"}\n        shape: {shape}\n        values: {array(values,f32)}\n'
    text+='*)\n'
    for run,_ in pipelines:
        text+=f'(* @assert:{run}:result\n{array(result,f32)}\n*)\n(* @assert:{run}:shape\n{output}\n*)\n'
    return text+source+'\n\n'

def dataset():
    yield 'global',[6],[1,2,3,4,5,6],None,False
    yield 'rows',[2,3],[1,2,3,4,5,6],1,False
    yield 'columns',[2,3],[1,2,3,4,5,6],0,False
    yield 'rank1-axis',[3],[1,2,3],0,False
    yield 'wrapping',[2,3],[0x7fffffff,1,0x80000000,0xffffffff,0xffffffff,2],1,False
    for i,(shape,axis) in enumerate([([0],None),([2,0],1),([0,3],0),([0,3],1),([2,0],0),([0,0],None)]):
        yield f'empty-{i}',shape,[],axis,False
    for name,shape,axis,values in [
        ('fractions',[2,3],1,[0x3f000000,0x3e800000,0xbf000000,0x3f800000,0x40000000,0x40400000]),
        ('cancellation',[4],None,[0x4b800000,0x3f800000,0x3f800000,0xcb800000]),
        ('exceptional',[3,2],1,[0x7f800000,0xff800000,0x7fc12345,0x3f800000,0x80000000,0x80000000]),
        ('ftz',[2,2],1,[1,0x80000001,0x00800000,0x80800001]),
        ('overflow',[2],None,[0x7f7fffff,0x7f7fffff]),
    ]: yield name,shape,values,axis,True

def invalid():
    out=''
    for name,source,message in [
        ('negative-axis','sum(a, axis=-1)','nonnegative integer'),
        ('axis-rank','sum(a, axis=2)','outside input rank'),
        ('axis-list','sum(a, axis=[0,1])','nonnegative integer'),
        ('duplicate-axis','sum(a, axis=0, axis=1)','nonnegative integer'),
        ('keepdims','sum(a, keepdims=true)','keepdims is unsupported'),
        ('missing-operand','sum()','requires an operand'),
        ('not-sum','a','requires sum('),
        ('unknown','sum(b)','currently requires one named input'),
    ]:
        out+=f'''(* @case {name} *)
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: sum-i32-cpu
    arguments:
      - name: a
        values: []
        shape: [0, 3]
*)
(* @assert:cpu:error
stage: frontend
message: "{message}"
*)
{source}

'''
    return out

def files():
    body=''.join(case(*d) for d in dataset())
    # Structural iterator and identity checks are device-independent.
    structure='''(* @test
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
'''
    return {'sum_cpu/semantics.ml':body,'sum_cpu/invalid.ml':invalid(),'sum_cpu/structure.ml':structure}

def anchors():
    require(tree([1,2,3],False)==6,'i32 tree anchor')
    require(tree([0x7fffffff,1],False)==0x80000000,'i32 wrap anchor')
    require(tree([0x4b800000,0x3f800000,0x3f800000,0xcb800000],True)==0x3f800000,'fixed tree cancellation anchor')
    require(tree([1,0x80000001],True)==0,'sum input FTZ anchor')
    require(sums([2,0],[],1,False)==([2],[0,0]),'empty reduction identity')
    require(sums([0,3],[],1,False)==([0],[]),'zero output count')

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--check',action='store_true');args=parser.parse_args()
    anchors();expected=files()
    if args.check:
        actual={str(p.relative_to(ROOT)) for folder in {Path(p).parts[0] for p in expected} for p in (ROOT/folder).rglob('*.ml')}
        require(actual==set(expected),'sum fixture file set differs')
        for path,text in expected.items():require((ROOT/path).read_text()==text,'sum fixture differs: '+path)
    else:
        for path,text in expected.items():(ROOT/path).parent.mkdir(exist_ok=True);(ROOT/path).write_text(text)
    print('Sum oracle anchors and',len(expected),'files verified')

if __name__=='__main__':main()
