import ast as pyast
from typing import Mapping, Optional

from .ast_nodes import Add, BinaryOp, Constant, Div, Exp, FuncCall, Mul, Neg, Node, Number, Pos, Sub, Sum, Var, X
from .errors import LoweringError
from .language import CONSTANT_NAMES, FunctionSpec


class ASTLowerer:
    def __init__(
        self,
        param_name: str,
        func_registry: Mapping[str, FunctionSpec],
        *,
        var_env: Optional[Mapping[str, float]] = None,
    ):
        self.param_name = param_name
        self.func_registry = func_registry
        self.var_env = dict(var_env or {})

    def lower(self, node: Node) -> pyast.AST:
        if isinstance(node, Number):
            return pyast.Constant(node.val)
        if isinstance(node, Constant):
            if node.name not in CONSTANT_NAMES:
                raise LoweringError(f"unknown constant: {node.name}", span=node.span)
            return pyast.Name(id=node.name, ctx=pyast.Load())
        if isinstance(node, X):
            return pyast.Name(id=self.param_name, ctx=pyast.Load())
        if isinstance(node, Var):
            if node.name in self.var_env:
                return pyast.Constant(self.var_env[node.name])
            raise LoweringError(f"unknown variable: {node.name}", span=node.span)
        if isinstance(node, Neg):
            return pyast.UnaryOp(op=pyast.USub(), operand=self.lower(node.node))
        if isinstance(node, Pos):
            return pyast.UnaryOp(op=pyast.UAdd(), operand=self.lower(node.node))
        if isinstance(node, Add):
            return pyast.BinOp(left=self.lower(node.l), op=pyast.Add(), right=self.lower(node.r))
        if isinstance(node, Sub):
            return pyast.BinOp(left=self.lower(node.l), op=pyast.Sub(), right=self.lower(node.r))
        if isinstance(node, Mul):
            return pyast.BinOp(left=self.lower(node.l), op=pyast.Mult(), right=self.lower(node.r))
        if isinstance(node, Div):
            return pyast.BinOp(left=self.lower(node.l), op=pyast.Div(), right=self.lower(node.r))
        if isinstance(node, Exp):
            return pyast.BinOp(left=self.lower(node.l), op=pyast.Pow(), right=self.lower(node.r))
        if isinstance(node, FuncCall):
            spec = self.func_registry.get(node.op)
            if spec is None:
                raise LoweringError(f"unknown function: {node.op}", span=node.span)
            if len(node.args) != spec.arity:
                raise LoweringError(
                    f"{node.op} expects {spec.arity} args, got {len(node.args)}",
                    span=node.span,
                )
            lowered_args = [self.lower(arg) for arg in node.args]
            return pyast.Call(
                func=pyast.Name(id=spec.target, ctx=pyast.Load()),
                args=lowered_args,
                keywords=[],
            )
        if isinstance(node, Sum):
            raise LoweringError("Sum nodes must be expanded before lowering", span=node.span)
        if isinstance(node, BinaryOp):
            raise LoweringError(f"unhandled binary op: {type(node).__name__}", span=node.span)
        raise LoweringError(f"unhandled node type: {type(node).__name__}", span=node.span)
