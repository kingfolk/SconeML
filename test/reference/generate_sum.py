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
        ('unknown','sum(b)','unknown tensor expression variable'),
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

def composed(name,expression,args,instructions,axis,f32,constants=None):
    shape=M4['shape_of'](args)
    mapped=[]
    if f32:
        shape,mapped=M4['evaluate'](args,instructions,constants or {},'r')
    else:
        for coord in product(*(range(n) for n in shape)):
            env=dict(constants or {})
            for key,dims,data in args:
                offset=0
                for extent,index in zip(dims,coord[-len(dims):]):offset=offset*extent+(0 if extent==1 else index)
                env[key]=data[offset]&MASK
            for dest,op,a,b in instructions:
                env[dest]={'add':lambda:env[a]+env[b],'sub':lambda:env[a]-env[b],'mul':lambda:env[a]*env[b]}[op]()&MASK
            mapped.append(env['r'])
    out_shape,result=sums(shape,mapped,axis,f32)
    dtype='f32' if f32 else 'i32'
    text=f'(* @case {name} *)\n(* @test\nversion: 1\ninput: source\nruns:\n'
    for run in ('cpu','metal'):
        text+=f'  - name: {run}\n    pipeline: sum-{dtype}-{run}\n'
        if f32:text+='    profile: f32-sum-tree256-ftz\n'
        text+='    arguments:\n'
        for key,dims,data in args:text+=f'      - name: {key}\n        dtype: {dtype}\n        shape: {dims}\n        values: {array(data,f32)}\n'
    text+='*)\n'
    for run in ('cpu','metal'):text+=f'(* @assert:{run}:result\n{array(result,f32)}\n*)\n(* @assert:{run}:shape\n{out_shape}\n*)\n'
    text+='(* @assert:metal:compare\ncpu\n*)\n'
    return text+'sum('+expression+(f', axis={axis}' if axis is not None else '')+')\n\n'

def composition_cases():
    text=''
    for f32 in (False,True):
        convert=(lambda vs:[M4['literal'](str(v)) for v in vs]) if f32 else (lambda vs:[v&MASK for v in vs])
        dtype='f32' if f32 else 'i32'
        args=[('a',[2,3],convert([1,2,3,4,5,6])),('b',[3],convert([10,20,30]))]
        for axis in (None,0,1):text+=composed(f'sample-{dtype}-{axis}','a * b',args,[('r','mul','a','b')],axis,f32)
        text+=composed(f'overlap-{dtype}','a * b',[args[0],('b',[3],convert([2,3,4]))],[('r','mul','a','b')],1,f32)
        text+=composed(f'three-{dtype}','a * b + c',args+[('c',[2,1],convert([1,-1]))],[('p','mul','a','b'),('r','add','p','c')],1,f32)
        text+=composed(f'empty-{dtype}','a * b',[('a',[2,0],[]),('b',[1],convert([1]))],[('r','mul','a','b')],1,f32)
        text+=composed(f'zero-output-{dtype}','a * b',[('a',[0,3],[]),('b',[3],convert([1,2,3]))],[('r','mul','a','b')],1,f32)
        text+=composed(f'unused-{dtype}','a + a',[('a',[3],convert([1,2,3])),('b',[2,1],convert([4,5]))],[('r','add','a','a')],0,f32)
    return text

def finite_bound(values):
    from fractions import Fraction
    normalized=[canonical(v) for v in values]
    require(all((v&0x7fffffff)<0x7f800000 for v in normalized),'bound requires finite inputs')
    exact=sum((M4['rational'](v) for v in normalized),Fraction())
    magnitude=sum((abs(M4['rational'](v)) for v in normalized),Fraction())
    result=tree(normalized,True)
    require((result&0x7fffffff)<0x7f800000,'bound case unexpectedly overflowed')
    n=len(normalized);q=0;depth=0
    while n:
        chunks=(n+255)//256;q+=255*chunks;depth+=8
        if chunks==1:break
        n=chunks
    u=Fraction(1,1<<24);eta=Fraction(1,1<<126)
    bound=depth*u/(1-depth*u)*magnitude+q*eta*(1+u)**depth
    require(abs(M4['rational'](result)-exact)<=bound,'size/magnitude/FTZ bound violated')

def acceptance():
    text=''
    for n in (2,3,255,256,257,511,512,513,1025,65537):
        values=[(i*2654435761)&MASK for i in range(n)]
        text+=case(f'global-i32-{n}',[n],values,None,False,[('cpu','sum-i32-cpu'),('metal','sum-i32-metal')])
    for axis,shape in ((0,[257,3]),(1,[3,257])):
        values=[(i*1103515245+12345)&MASK for i in range(prod(shape))]
        text+=case(f'strided-i32-axis{axis}',shape,values,axis,False,[('cpu','sum-i32-cpu'),('metal','sum-i32-metal')])
    for n in (1,2,3,255,256,257,513,1025,4097):
        state=0x5c0e0005;values=[]
        for i in range(n):
            state=(1664525*state+1013904223)&MASK
            values.append((state&0x807fffff)|((100+(state>>24)%41)<<23))
        finite_bound(values)
        text+=case(f'finite-f32-{n}',[n],values,None,True,[('cpu','sum-f32-cpu'),('metal','sum-f32-metal')])
    for name,values in [
        ('cancel-cross-group',[0x4b800000,0x3f800000,0x3f800000,0xcb800000]+[0]*252+[0x3f800000]),
        ('nan-cross-group',[0x7f800000]+[0]*255+[0xff800000]),
        ('overflow-tree',[0x7f7fffff,0x7f7fffff,0xff7fffff,0xff7fffff]),
        ('negative-zero-full-group',[0x80000000]*256),
        ('tiny-inputs',[0x007fffff,0x807fffff]*257),
        ('signaling-nan-tail',[0]*512+[0x7f800001]),
    ]:
        text+=case(name,[len(values)],values,None,True,[('cpu','sum-f32-cpu'),('metal','sum-f32-metal')])
    for f32 in (False,True):
        dtype='f32' if f32 else 'i32';convert=(lambda x:M4['literal'](str(x))) if f32 else (lambda x:x&MASK)
        args=[('a',[257,3],[convert(i%13-6) for i in range(771)]),('b',[3],[convert(1),convert(2),convert(3)]),('c',[257,1],[convert(i%5) for i in range(257)])]
        for axis in (0,1,None):text+=composed(f'composed-{dtype}-{axis}','a * b + c',args,[('p','mul','a','b'),('r','add','p','c')],axis,f32)
        text+=composed(f'empty-unused-{dtype}','a + a',[('a',[1,3],[convert(1)]*3),('b',[0,3],[])],[('r','add','a','a')],0,f32)
    return text

def sum_goldens():
    text=''
    for dtype in ('i32','f32'):
        expected=(Path(__file__).parent/'msl'/f'sum-{dtype}.msl').read_text()
        profile='    profile: f32-sum-tree256-ftz\n' if dtype=='f32' else ''
        text+=f'(* @case full-{dtype}-shader *)\n(* @test\nversion: 1\ninput: source\nruns:\n  - name: shader\n    pipeline: sum-{dtype}-msl\n'+profile+f'    arguments:\n      - name: a\n        dtype: {dtype}\n        values: [1,2,3,4,5,6]\n        shape: [2,3]\n*)\n'
        text+='(* @assert:shader:text-equal\n'+expected+'*)\n(* @assert:shader:entrypoint\nsum_tree\n*)\nsum(a, axis=1)\n\n'
    return text

def acceptance_errors():
    text=''
    empty=[('a',[0,3],[])]
    for name,source,args,message,f32 in [
        ('bad-storage','sum(a, axis=1)',[('a',[0,3],[1])],'storage count does not match shape product',False),
        ('bad-rank','sum(a)',[('a',[0,1,1],[])],'rank must be 1 or 2',False),
        ('duplicate-name','sum(a)',empty+empty,'duplicate tensor argument name',False),
        ('no-inputs','sum(1)',[],'between one and three',False),
        ('bad-broadcast','sum(a + b, axis=1)',empty+[('b',[2],[1,2])],'incompatible shaped tensor broadcast',False),
        ('reserved-name','sum(let)',[('let',[0],[])],'invalid tensor argument name',False),
        ('missing-map-operand','sum(a +, axis=1)',empty,'missing an operand',False),
        ('fractional-axis','sum(a, axis=1.0)',empty,'nonnegative integer',False),
        ('literal-overflow-empty','sum(a * 1e100, axis=1)',empty,'outside finite binary32 range',True),
        ('nested-reduction','sum(sum(a))',empty,'expected +, - or *',False),
    ]:
        dtype='f32' if f32 else 'i32'
        text+=f'(* @case {name} *)\n(* @test\nversion: 1\ninput: source\nruns:\n'
        for run in ('cpu','metal'):
            text+=f'  - name: {run}\n    pipeline: sum-{dtype}-{run}\n'
            if f32:text+='    profile: f32-sum-tree256-ftz\n'
            text+='    arguments:'+(' []\n' if not args else '\n')
            for key,shape,data in args:text+=f'      - name: {key}\n        dtype: {dtype}\n        shape: {shape}\n        values: {array(data,f32)}\n'
        text+='*)\n'
        for run in ('cpu','metal'):text+=f'(* @assert:{run}:error\nstage: frontend\nmessage: "{message}"\n*)\n'
        text+=source+'\n\n'
    return text

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
    gpu=''
    extra=list(dataset())
    for f32 in (False,True):
        for n in (1,255,256,257,513):
            values=[0x3f800000]*n if f32 else [(i*65537)&MASK for i in range(n)]
            extra.append((f'grid-{n}-'+('f32' if f32 else 'i32'),[n],values,None,f32))
    for name,shape,data,axis,f32 in extra:
        dtype='f32' if f32 else 'i32'
        gpu+=case(name,shape,data,axis,f32,[('cpu',f'sum-{dtype}-cpu'),('metal',f'sum-{dtype}-metal')])
    gpu_invalid=invalid().replace('sum-i32-cpu','sum-i32-metal')
    msl=structure.replace('sum-i32-ir','sum-i32-msl')
    start=msl.index('(* @assert:ir:ir-check');end=msl.index('*)',start)+2
    msl=msl[:start]+'''(* @assert:ir:text-check
CHECK: threadgroup Element scratch[256]
CHECK: k < length ?
CHECK: threadgroup_barrier(mem_flags::mem_threadgroup)
CHECK: stride *= 2u
CHECK: scratch[lane] = sum_add(scratch[lane], scratch[lane + stride])
CHECK: threadgroup_barrier(mem_flags::mem_threadgroup)
CHECK: output[group] = scratch[0]
*)'''+msl[end:]
    return {'sum_cpu/semantics.ml':body,'sum_cpu/invalid.ml':invalid(),'sum_cpu/structure.ml':structure,
            'sum_gpu/semantics.ml':gpu,'sum_gpu/invalid.ml':gpu_invalid,'sum_gpu/structure.ml':msl,'map_sum/semantics.ml':composition_cases(),'acceptance_m5/numerics.ml':acceptance(),'acceptance_m5/shaders.ml':sum_goldens(),'acceptance_m5/invalid.ml':acceptance_errors()}

def anchors():
    require(tree([1,2,3],False)==6,'i32 tree anchor')
    require(tree([0x7fffffff,1],False)==0x80000000,'i32 wrap anchor')
    require(tree([0x4b800000,0x3f800000,0x3f800000,0xcb800000],True)==0x3f800000,'fixed tree cancellation anchor')
    require(tree([1,0x80000001],True)==0,'sum input FTZ anchor')
    require(sums([2,0],[],1,False)==([2],[0,0]),'empty reduction identity')
    require(sums([0,3],[],1,False)==([0],[]),'zero output count')
    finite_bound([0x4b800000,0x3f800000,0x3f800000,0xcb800000])

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
