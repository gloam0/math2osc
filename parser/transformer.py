from numpy import pi, e
from lark import Lark, Transformer, v_args, Token
from .ast_nodes import *

parser = Lark.open(
    "grammar.lark",
    parser="lalr",
    propagate_positions=True,
    rel_to=__file__,
)

def get_span(meta):
    return meta.start_pos, meta.end_pos

class ParseError(Exception):
    def __init__(self, message: str, span=None):
        super().__init__(message)
        self.span = span

@v_args(meta=True)
class AST(Transformer):
    def num(self, meta, items):
        if len(items) != 1:
            raise ParseError(f"number expected 1 arg, got {len(items)} args:\n{items}",
                             span=get_span(meta))
        return Number(span=get_span(meta), val=float(items[0]))

    def constant(self, meta, items):
        constants = ["pi", "tau", "e", "sr", "dt"]
        if len(items) != 1:
            raise ParseError(f"constant expected 1 arg, got {len(items)} args:\n{items}",
                             span=get_span(meta))
        tok = str(items[0])
        if tok not in constants:
            raise ParseError(f"constant was recognized constant: {tok}",
                             span=get_span(meta))
        return Constant(span=get_span(meta), name=tok)

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

    def args(self, meta, items):
        return list(items)

    def func_call(self, meta, items):
        if len(items) != 2:
            raise ParseError(f"func_call expected 2 args, got {len(items)} args:\n{items}",
                             span=get_span(meta))
        return FuncCall(span=get_span(meta), op=str(items[0]), args=items[1])
