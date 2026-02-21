from lark import Lark, Token, Transformer, Tree, v_args

from .ast_nodes import Add, Constant, Div, Exp, FuncCall, Mul, Neg, Number, Pos, Sub, Sum, Var, X
from .errors import ParseError
from .language import CONSTANT_NAMES, PARAM_SYMBOL, STEP_KEYWORD, SUM_KEYWORD

parser = Lark.open(
    "grammar.lark",
    parser="lalr",
    lexer="contextual",
    propagate_positions=True,
    rel_to=__file__,
)

def get_span(meta):
    return meta.start_pos, meta.end_pos

@v_args(meta=True)
class _AST(Transformer):
    def num(self, meta, items):
        if len(items) != 1:
            raise ParseError(f"number expected 1 arg, got {len(items)} args:\n{items}",
                             span=get_span(meta))
        return Number(span=get_span(meta), val=float(items[0]))

    def symbol(self, meta, items):
        if len(items) != 1:
            raise ParseError(f"symbol expected 1 arg, got {len(items)} args:\n{items}",
                             span=get_span(meta))
        name = str(items[0])
        if name == PARAM_SYMBOL:
            return X(span=get_span(meta))
        if name in CONSTANT_NAMES:
            return Constant(span=get_span(meta), name=name)
        return Var(span=get_span(meta), name=name)

    def neg(self, meta, items):
        if len(items) != 1:
            raise ParseError(f"neg expected 1 arg, got {len(items)} args:\n{items}",
                             span=get_span(meta))
        return Neg(span=get_span(meta), node=items[0])

    def pos(self, meta, items):
        if len(items) != 1:
            raise ParseError(f"pos expected 1 arg, got {len(items)} args:\n{items}",
                             span=get_span(meta))
        return Pos(span=get_span(meta), node=items[0])

    def add(self, meta, items):
        if len(items) != 2:
            raise ParseError(f"add expected 2 args, got {len(items)} args:\n{items}",
                             span=get_span(meta))
        return Add(span=get_span(meta), l=items[0], r=items[1])

    def sub(self, meta, items):
        if len(items) != 2:
            raise ParseError(f"sub expected 2 args, got {len(items)} args:\n{items}",
                             span=get_span(meta))
        return Sub(span=get_span(meta), l=items[0], r=items[1])

    def mul(self, meta, items):
        if len(items) != 2:
            raise ParseError(f"mul expected 2 args, got {len(items)} args:\n{items}",
                             span=get_span(meta))
        return Mul(span=get_span(meta), l=items[0], r=items[1])

    def div(self, meta, items):
        if len(items) != 2:
            raise ParseError(f"div expected 2 args, got {len(items)} args:\n{items}",
                             span=get_span(meta))
        return Div(span=get_span(meta), l=items[0], r=items[1])

    def exp(self, meta, items):
        if len(items) != 2:
            raise ParseError(f"exp expected 2 args, got {len(items)} args:\n{items}",
                             span=get_span(meta))
        return Exp(span=get_span(meta), l=items[0], r=items[1])

    def sum_range(self, meta, items):
        if len(items) != 3:
            raise ParseError(f"sum_range expected 3 args, got {len(items)} args:\n{items}",
                             span=get_span(meta))
        var_name = str(items[0])
        if var_name == PARAM_SYMBOL:
            raise ParseError(f"sum loop variable cannot be '{PARAM_SYMBOL}'", span=get_span(meta))
        if var_name in CONSTANT_NAMES:
            raise ParseError(f"sum loop variable cannot shadow constant '{var_name}'", span=get_span(meta))
        return var_name, items[1], items[2]

    def sum_step(self, meta, items):
        if len(items) != 2 or not isinstance(items[0], Token):
            raise ParseError(f"sum_step expected 2 args, got {len(items)} args:\n{items}",
                             span=get_span(meta))
        step_name = str(items[0])
        if step_name != STEP_KEYWORD:
            raise ParseError(f"expected '{STEP_KEYWORD}=...' in sum, got '{step_name}=...'",
                             span=get_span(meta))
        return items[1]

    def args(self, meta, items):
        return list(items)

    def func_call(self, meta, items):
        if len(items) != 2:
            raise ParseError(f"func_call expected 2 args, got {len(items)} args:\n{items}",
                             span=get_span(meta))
        return FuncCall(span=get_span(meta), op=str(items[0]), args=items[1])

    def sum_call(self, meta, items):
        if len(items) not in (3, 4):
            raise ParseError(f"sum_call expected 3 or 4 args, got {len(items)} args:\n{items}",
                             span=get_span(meta))
        if not isinstance(items[0], Token):
            raise ParseError("sum_call missing function name token", span=get_span(meta))
        sum_name = str(items[0])
        if sum_name != SUM_KEYWORD:
            raise ParseError(f"unknown function with sum syntax: {sum_name}",
                             span=get_span(meta))
        var_name, lo, hi = items[1]
        if len(items) == 3:
            step = None
            body = items[2]
        else:
            step = items[2]
            body = items[3]
        return Sum(span=get_span(meta), var=var_name, lo=lo, hi=hi, step=step, expr=body)


_ast = _AST()
def do_transform(tree: Tree):
    return _ast.transform(tree)
