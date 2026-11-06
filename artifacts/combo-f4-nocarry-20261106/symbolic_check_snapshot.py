"""Check floor-division splitting over finite index domains; no NPU launch required."""
import ast
import hashlib
import itertools
import json
from pathlib import Path
import random
from types import SimpleNamespace
import sys

import sympy
import torch
import torch_npu
import torch_npu._inductor
from torch._inductor.sizevars import SizeVarAllocator
from torch._inductor.virtualized import V
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch_npu._inductor.codegen import ir

source = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(ir.__file__)
namespace = dict(vars(ir))
from torch.utils._sympy.functions import PythonMod
from torch.utils._sympy.value_ranges import bound_sympy, ValueRangeError
namespace.update(PythonMod=PythonMod, bound_sympy=bound_sympy, ValueRangeError=ValueRangeError)
selected = {'_floor_div_terms_do_not_carry', 'eliminate_zero_term', 'split_expression'}
for node in ast.parse(source.read_text()).body:
    if isinstance(node, ast.FunctionDef) and node.name in selected:
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)

x, y, z = sympy.symbols('x y z', integer=True, nonnegative=True)
graph = SimpleNamespace(sizevars=SizeVarAllocator(ShapeEnv()))
summary = {'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'examples': [],
           'random_domains': 0, 'checked_points': 0, 'proved_domains': 0}

def check(coefficients, lengths, divisor, label=None):
    numerator = sum(c * s for c, s in zip(coefficients[:3], (x, y, z))) + coefficients[3]
    kernel = SimpleNamespace(range_tree_nodes={s: SimpleNamespace(length=sympy.Integer(n))
                                               for s, n in zip((x, y, z), lengths)},
                             range_tree_nodes_removed={})
    terms = sympy.Add.make_args(numerator)
    with V.set_graph_handler(graph), V.set_kernel_handler(kernel):
        proven = namespace['_floor_div_terms_do_not_carry'](terms, sympy.Integer(divisor))
        result = namespace['split_expression'](ir.FloorDiv(numerator, divisor))
    code = compile(str(result), '<split-expression>', 'eval')
    for values in itertools.product(*(range(n) for n in lengths)):
        actual = eval(code, {'__builtins__': {}}, dict(zip(('x','y','z'), values)))
        expected = (sum(c*v for c,v in zip(coefficients[:3], values)) + coefficients[3]) // divisor
        assert actual == expected, (numerator, divisor, result, values, actual, expected)
        if proven:
            split = sum(int(term.subs(dict(zip((x,y,z), values)))) // divisor for term in terms)
            assert split == expected, (terms, values, divisor)
        summary['checked_points'] += 1
    summary['proved_domains'] += int(proven)
    if label:
        summary['examples'].append({'label': label, 'numerator': str(numerator), 'lengths': lengths,
                                    'divisor': divisor, 'no_carry_proven': proven, 'result': str(result)})
    return proven

assert check((10, 1, 0, 0), (2, 6, 1), 8, 'safe non-divisible terms')
assert not check((10, 1, 0, 0), (2, 7, 1), 8, 'carry at equality')
assert not check((8, 0, 0, 10), (6, 1, 1), 3, 'pooling boundary must remain combined')
assert not check((2, 2, 2, 0), (2, 2, 2), 5, 'three remainders carry together')
check((-6, 1, 0, -1), (3, 4, 1), 8, 'signed terms use Python modulo')
with V.set_graph_handler(graph), V.set_kernel_handler(SimpleNamespace(range_tree_nodes={}, range_tree_nodes_removed={})):
    assert not namespace['_floor_div_terms_do_not_carry']((3*x, y), sympy.Integer(8))
    assert not namespace['_floor_div_terms_do_not_carry']((x, y), sympy.Integer(-8))
    opaque = sympy.Function('opaque', integer=True)
    assert not namespace['_floor_div_terms_do_not_carry']((opaque(x), y), sympy.Integer(8))
rng = random.Random(20261106)
for _ in range(500):
    check(tuple(rng.randint(-16, 24) for _ in range(4)), tuple(rng.randint(1, 5) for _ in range(3)),
          rng.randint(2, 16))
    summary['random_domains'] += 1
summary['result'] = 'passed'
print(json.dumps(summary, indent=2))
