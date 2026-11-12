#!/usr/bin/env python3
"""Independent binary32 oracle and self-contained M4 acceptance fixtures.

Uses exact Python integers/Fraction, never host float/double arithmetic, LLVM,
production parser/indexing, compiler execution, or actual backend output.
Source text and oracle instruction lists are deliberately separate. --check is
read-only, checks anchors and the complete generated file set, and works with -O.
"""
from fractions import Fraction
from itertools import product
from math import prod
from pathlib import Path
import argparse
import sys

sys.dont_write_bytecode = True
SIGN, INF, NAN = 0x80000000, 0x7F800000, 0x7FC00000
MIN_NORMAL = Fraction(1, 1 << 126)
SEED = 0x5C0E0004
ROOT = Path(__file__).resolve().parents[1] / "acceptance_m4"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def magnitude(bits):
    return bits & 0x7FFFFFFF


def ftz(bits):
    return bits & SIGN if magnitude(bits) < 0x00800000 else bits


def rational(bits):
    """Exact signed rational for a finite binary32 bit pattern."""
    exponent, significand = (bits >> 23) & 255, bits & 0x7FFFFF
    require(exponent != 255, "nonfinite rational")
    if exponent:
        significand |= 1 << 23
        exponent -= 150
    else:
        exponent = -149
    value = Fraction(significand) * (Fraction(2) ** exponent)
    return -value if bits & SIGN else value


def nearest_even(n, d):
    q, r = divmod(n, d)
    return q + (2 * r > d or (2 * r == d and q % 2 == 1))


def rounded(value, negative_zero=False):
    """RNE to binary32, including gradual rounding for decimal conversion."""
    sign = SIGN if value < 0 or (not value and negative_zero) else 0
    value = abs(value)
    if not value:
        return sign
    n, d = value.numerator, value.denominator
    exponent = n.bit_length() - d.bit_length()
    if Fraction(2) ** exponent > value:
        exponent -= 1
    scale = 23 - max(exponent, -126)
    if scale >= 0:
        sig = nearest_even(n << scale, d)
    else:
        sig = nearest_even(n, d << -scale)
    if exponent < -126:
        return sign | sig  # May round up to exactly minimum normal.
    if sig == 1 << 24:
        sig >>= 1
        exponent += 1
    if exponent > 127:
        return sign | INF
    return sign | ((exponent + 127) << 23) | (sig - (1 << 23))


def literal(text):
    result = rounded(Fraction(text), text.startswith("-"))
    require(magnitude(result) < INF, "oracle literal overflow")
    return ftz(result)


def arithmetic(op, left, right):
    """f32-ftz: signed input FTZ, exact op, pre-round tininess, RNE."""
    a, b = ftz(left), ftz(right)
    if magnitude(a) > INF or magnitude(b) > INF:
        return NAN
    if op == "sub":
        b ^= SIGN
        op = "add"
    if op == "mul":
        sign = (a ^ b) & SIGN
        if magnitude(a) == INF or magnitude(b) == INF:
            if not magnitude(a) or not magnitude(b):
                return NAN
            return sign | INF
        exact = rational(a) * rational(b)
        if abs(exact) < MIN_NORMAL:
            return sign
        return rounded(exact)
    require(op == "add", "unknown oracle operation")
    if magnitude(a) == INF:
        return NAN if magnitude(b) == INF and ((a ^ b) & SIGN) else a
    if magnitude(b) == INF:
        return b
    exact = rational(a) + rational(b)
    if not exact:
        return (a & b & SIGN) if not magnitude(a) and not magnitude(b) else 0
    if abs(exact) < MIN_NORMAL:
        return SIGN if exact < 0 else 0
    return rounded(exact)


def anchors():
    # Hand-derived results, including separate versus fused multiply-add and
    # tininess before rounding. No Python assert: -O must retain every check.
    for op, a, b, expected in [
        ("add", 0x3F800000, 0x33800000, 0x3F800000),
        ("add", 0x3F800001, 0x33800000, 0x3F800002),
        ("mul", 0x3F800001, 0x3F7FFFFE, 0x3F800000),
        ("sub", 0x3F800000, 0x3F800000, 0),
        ("mul", 0x00800000, 0x3F7FFFFF, 0),
        ("mul", 0x80800000, 0x3F7FFFFF, SIGN),
        ("mul", 0x00800001, 0x3F7FFFFE, 0),
        ("mul", 0x00800000, 0x3F800000, 0x00800000),
        ("mul", 0x00000001, 0x7F7FFFFF, 0),
        ("mul", 0x80000001, 0x7F7FFFFF, SIGN),
        ("sub", 0x00800001, 0x00800000, 0),
        ("sub", 0x80800001, 0x80800000, SIGN),
        ("mul", 0x7F7FFFFF, 0x40000000, INF),
        ("mul", 0xFF7FFFFF, 0x40000000, SIGN | INF),
        ("mul", INF, 0, NAN),
        ("add", INF, SIGN | INF, NAN),
        ("add", SIGN, SIGN, SIGN),
        ("add", SIGN, 0, 0),
        ("sub", SIGN, 0, SIGN),
        ("sub", SIGN, SIGN, 0),
        ("mul", SIGN, 0x3F800000, SIGN),
    ]:
        require(arithmetic(op, a, b) == expected, f"oracle anchor: {op} {a:08x} {b:08x}")
    require(literal("0.1") == 0x3DCCCCCD, "decimal rounding anchor")
    require(literal("1.000000059604644775390625") == 0x3F800000, "decimal even tie")
    require(literal("1.000000178813934326171875") == 0x3F800002, "decimal odd tie")
    require(literal("-0.0") == SIGN and literal("-1e-40") == SIGN, "literal signed FTZ")
    require(rounded(MIN_NORMAL - Fraction(1, 1 << 150)) == 0x00800000, "pre-round tininess distinction")
    # Exhaust the zero/Inf/NaN sign combinations and both operands' subnormal
    # normalization, without assuming payload propagation.
    for a in [0, SIGN, INF, SIGN | INF, NAN, NAN | SIGN, 1, SIGN | 1]:
        for b in [0, SIGN, INF, SIGN | INF, NAN, 0x3F800000]:
            for op in ["add", "sub", "mul"]:
                r = arithmetic(op, a, b)
                require(magnitude(r) == 0 or magnitude(r) >= 0x00800000, "oracle emitted subnormal")


def shape_of(arguments):
    rank = max(len(s) for _, s, _ in arguments)
    result = []
    for axis in range(rank):
        extents = [([1] * (rank - len(s)) + s)[axis] for _, s, _ in arguments]
        non_singletons = set(n for n in extents if n != 1)
        require(len(non_singletons) <= 1, "incompatible oracle shapes")
        result.append(next(iter(non_singletons)) if non_singletons else 1)
    return result


def evaluate(arguments, instructions, constants, result):
    shape = shape_of(arguments)
    values = []
    # Logical-coordinate cartesian product; no production flattened div/rem
    # or affine map implementation is reused.
    for coord in product(*(range(n) for n in shape)):
        environment = dict(constants)
        for name, dims, data in arguments:
            selected = coord[-len(dims):]
            offset = 0
            for extent, axis in zip(dims, selected):
                offset = offset * extent + (0 if extent == 1 else axis)
            environment[name] = ftz(data[offset])
        for destination, op, left, right in instructions:
            environment[destination] = arithmetic(op, environment[left], environment[right])
        values.append(environment[result])
    return shape, values


def bits_list(values):
    return "[" + ", ".join(f"0x{n:08x}" for n in values) + "]"


def fixture(name, source, arguments, instructions=(), constants=None,
            result="r", width=None):
    shape, values = evaluate(arguments, instructions, constants or {}, result)
    text = f"(* @case {name} *)\n(* @test\nversion: 1\ninput: source\nruns:\n"
    for run, pipeline in [("cpu", "shaped-f32-cpu"), ("metal", "shaped-f32-metal")]:
        text += f"  - name: {run}\n    pipeline: {pipeline}\n    profile: f32-ftz\n"
        if run == "metal" and width:
            text += f"    threads-per-group: {width}\n"
        text += "    arguments:\n"
        for name, dims, data in arguments:
            require(len(data) == prod(dims), "oracle input count")
            text += f"      - name: {name}\n        dtype: f32\n        shape: {dims}\n        values: {bits_list(data)}\n"
    text += "*)\n"
    for run in ["cpu", "metal"]:
        text += f"(* @assert:{run}:result\n{bits_list(values)}\n*)\n(* @assert:{run}:shape\n{shape}\n*)\n"
    text += "(* @assert:metal:compare\ncpu\n*)\n" + source + "\n\n"
    return text


def stream(count, salt):
    state = SEED ^ salt
    values = []
    for _ in range(count):
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        values.append(state)
    return values


def files():
    out = {}
    t = ""
    # Every lane has an independent expectation; CPU/Metal equality alone is
    # never sufficient to accept a numerical run.
    for name, op, a, b in [
        ("round-even", "add", [0x3F800000, 0x3F800001], [0x33800000]*2),
        ("pre-round-tininess", "mul", [0x00800000,0x80800000,0x00800001,0x00800000], [0x3F7FFFFF,0x3F7FFFFF,0x3F7FFFFE,0x3F800000]),
        ("normal-boundaries", "mul", [0x00800000,0x00800001,0x00FFFFFF,0x80800000], [0x3F800000,0x3F800000,0x3F000000,0x3F800001]),
        ("cancel-tiny", "sub", [0x00800001,0x80800001,0x3F800001,0xBF800001], [0x00800000,0x80800000,0x3F800000,0xBF800000]),
        ("overflow-signs", "mul", [0x7F7FFFFF,0xFF7FFFFF,0x7F7FFFFF,0xFF7FFFFF], [0x40000000,0x40000000,0xBF800000,0xBF800000]),
    ]:
        symbol = {"add":"+","sub":"-","mul":"*"}[op]
        t += fixture(name, f"a {symbol} b", [("a",[len(a)],a),("b",[len(b)],b)], [("r",op,"a","b")], width=4)
    specials = [0,SIGN,1,SIGN|1,0x007FFFFF,0x807FFFFF,0x00800000,0x80800000,0x3F800000,0xBF800000,INF,SIGN|INF,NAN,0xFFC12345,0x7F800001]
    for op, symbol in [("add","+"),("sub","-"),("mul","*")]:
        # Full cross product includes signaling NaNs and signed zeros/Inf.
        a = [x for x in specials for _ in specials]
        b = specials * len(specials)
        t += fixture("exceptional-"+op, f"a {symbol} b", [("a",[len(a)],a),("b",[len(b)],b)], [("r",op,"a","b")], width=32)
    t += fixture("identity-bits", "a", [("a",[len(specials)],specials)], result="a")
    t += fixture("noncontracted", "a * b + c", [("a",[1],[0x3F800001]),("b",[1],[0x3F7FFFFE]),("c",[1],[0xBF800000])], [("m","mul","a","b"),("r","add","m","c")])
    t += fixture("intermediate-ftz", "a * 0.5 * 16777216", [("a",[4],[0x00800000,0x80800000,0x00800001,0x80800001])], [("m","mul","a","half"),("r","mul","m","large")], {"half":literal("0.5"),"large":literal("16777216")})
    for i, spelling in enumerate(["0.1", "-0.0", "-1e-40", "1.000000059604644775390625", "1.000000178813934326171875", "3.4028234663852886e38", ".5", "+0.5", "1.1754942807573643e-38", "-1.1754942807573643e-38"]):
        t += fixture("constant-"+str(i), spelling, [("a",[2,3],stream(6,1))], constants={"k":literal(spelling)}, result="k")
    t += fixture("repeated-products", "a * a + a * a", [("a",[9],stream(9,9))], [("p","mul","a","a"),("q","mul","a","a"),("r","add","p","q")])
    out["arithmetic.ml"] = t
    t = ""
    for i, shapes in enumerate([
        [[2,3],[3],[2,1]], [[3],[2,1],[1,1]], [[1],[3,1],[1,5]],
        [[1,7],[7],[1]], [[7,1],[1],[1,1]], [[3,5],[1,5],[3,1]],
        [[5,3],[3],[5,1]], [[2,17],[1,17],[2,1]], [[1,6],[6],[1]],
        [[6],[1,6],[1,1]], [[1,1],[1],[1,1]], [[2,3],[2,3],[2,3]],
    ]):
        args = [(name,shape,stream(prod(shape),i*3+j)) for j,(name,shape) in enumerate(zip(["a","b","c"],shapes))]
        t += fixture(f"shape-{i}", "a * b + c - a", args, [("p","mul","a","b"),("q","add","p","c"),("r","sub","q","a")], width=4)
    # Unused declared inputs still determine the common output shape.
    t += fixture("unused-input", "a + 0.25", [("a",[3],[literal("0.5"),literal("1.5"),literal("-0.5")]),("b",[2,1],stream(2,22))], [("r","add","a","k")], {"k":literal("0.25")})
    out["broadcast.ml"] = t
    t = ""
    for width in [4,32,256]:
        for i,n in enumerate([1,width-1,width,width+1,2*width+3]):
            shapes = [[n],[1],[n]] if i%2==0 else [[1,n],[n],[1,1]]
            args=[(name,shape,stream(prod(shape),width+j)) for j,(name,shape) in enumerate(zip(["a","b","c"],shapes))]
            t += fixture(f"w{width}-n{n}","a * 0.1 + b - c",args,[("p","mul","a","k"),("q","add","p","b"),("r","sub","q","c")],{"k":literal("0.1")},width=width)
    out["grid.ml"] = t
    t = ""
    for i, shapes in enumerate([[[0],[1]],[[0,3],[3]],[[2,0],[1,0]],[[0,0],[1]],[[0,1],[1,3]],[[2,0],[1]]]):
        args=[(name,shape,stream(prod(shape),i+j)) for j,(name,shape) in enumerate(zip(["a","b"],shapes))]
        t += fixture(f"empty-{i}","a * 0.5 + b",args,[("p","mul","a","k"),("r","add","p","b")],{"k":literal("0.5")},width=32)
    out["empty.ml"] = t
    t = ""
    for i, source, instructions, constants, result in [
        (0,"a + b",[("r","add","a","b")],{},"r"),
        (1,"a - b",[("r","sub","a","b")],{},"r"),
        (2,"a * b",[("r","mul","a","b")],{},"r"),
        (3,"a * b + c - a",[("p","mul","a","b"),("q","add","p","c"),("r","sub","q","a")],{},"r"),
        (4,"a * 0.1 + b",[("p","mul","a","k"),("r","add","p","b")],{"k":literal("0.1")},"r"),
    ]:
        args=[(name,[257],stream(257,i*17+j)) for j,name in enumerate(["a","b","c"])]
        t += fixture(f"seeded-{i}",source,args,instructions,constants,result,width=256)
    out["magnitude.ml"] = t
    # Complete independently hand-authored shader, never copied from a backend.
    golden = (Path(__file__).resolve().parent / "msl" / "f32-add.msl").read_text()
    text = fixture("f32-add-golden", "a + 0.5", [("a",[3],[literal("1"),literal("0.5"),SIGN])],
                   [("r","add","a","k")], {"k":literal("0.5")}, width=4)
    device = """  - name: device
    pipeline: shaped-f32-msl
    profile: f32-ftz
    arguments:
      - name: a
        dtype: f32
        shape: [3]
        values: [1.0, 0.5, -0.0]
"""
    first_end=text.index("*)",text.index("(* @test"))
    text=text[:first_end]+device+text[first_end:]
    marker="(* @assert:device:text-equal\n"+golden+"*)\n"
    marker+="(* @assert:device:source-unchanged\ntrue\n*)\n"
    marker+="(* @assert:device:entrypoint\ntensor_map\n*)\n"
    text=text.replace("a + 0.5\n",marker+"a + 0.5\n")
    out["structure.ml"]=text
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check",action="store_true")
    args=parser.parse_args()
    anchors()
    expected=files()
    if args.check:
        require(ROOT.is_dir(),"missing M4 fixtures")
        actual={p.name for p in ROOT.iterdir() if p.is_file()}
        require(actual==set(expected),f"M4 fixture set differs: {actual ^ set(expected)}")
        for name,contents in expected.items():
            require((ROOT/name).read_text()==contents,f"M4 fixture differs: {name}")
    else:
        ROOT.mkdir(exist_ok=True)
        for name,contents in expected.items():
            (ROOT/name).write_text(contents)
    count=sum(text.count("(* @case ") for text in expected.values())
    runs=sum(text.count("    pipeline:") for text in expected.values())
    print(f"M4 rational-oracle anchors and {len(expected)} files / {count} cases / {runs} runs verified")

if __name__=="__main__":
    main()
