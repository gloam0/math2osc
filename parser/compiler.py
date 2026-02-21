import ast as pyast
from typing import Callable, Mapping, Optional

from lark.exceptions import UnexpectedInput, VisitError

from .ast_nodes import Node
from .errors import DSLCompileError, LoweringError, ParseError
from .expander import expand_sums
from .language import DEFAULT_FUNC_REGISTRY, FunctionSpec
from .lowerer import ASTLowerer
from .runtime_env import build_eval_env
from .transformer import do_transform, parser


def parse_dsl(expr: str) -> Node:
    try:
        tree = parser.parse(expr)
    except UnexpectedInput as exc:
        pos = getattr(exc, "pos_in_stream", None)
        span = (pos, pos + 1) if isinstance(pos, int) else None
        raise ParseError(str(exc), span=span) from exc
    try:
        return do_transform(tree)
    except VisitError as exc:
        if isinstance(exc.orig_exc, DSLCompileError):
            raise exc.orig_exc from exc
        raise


def node_to_ast(
    node: Node,
    *,
    param_name: str = "x",
    func_registry: Mapping[str, FunctionSpec] = DEFAULT_FUNC_REGISTRY,
    sr: Optional[float] = None,
    dt: Optional[float] = None,
) -> pyast.AST:
    expanded = expand_sums(
        node,
        func_registry=func_registry,
        sr=sr,
        dt=dt,
    )
    return ASTLowerer(
        param_name=param_name,
        func_registry=dict(func_registry),
    ).lower(expanded)


def node_to_expanded_str(
    node: Node,
    *,
    param_name: str = "x",
    func_registry: Mapping[str, FunctionSpec] = DEFAULT_FUNC_REGISTRY,
    sr: Optional[float] = None,
    dt: Optional[float] = None,
) -> str:
    expr = node_to_ast(
        node,
        param_name=param_name,
        func_registry=func_registry,
        sr=sr,
        dt=dt,
    )
    pyast.fix_missing_locations(expr)
    try:
        return pyast.unparse(expr)
    except AttributeError as exc:
        raise LoweringError("ast.unparse is unavailable in this Python version") from exc


def build_callable(
    node: Node,
    *,
    param_name: str = "x",
    func_registry: Mapping[str, FunctionSpec] = DEFAULT_FUNC_REGISTRY,
    sr: Optional[float] = None,
    dt: Optional[float] = None,
    extra_env: Optional[Mapping[str, object]] = None,
) -> Callable:
    expr = node_to_ast(node, param_name=param_name, func_registry=func_registry, sr=sr, dt=dt)
    lam = pyast.Lambda(
        args=pyast.arguments(
            posonlyargs=[],
            args=[pyast.arg(arg=param_name)],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=expr,
    )
    wrapped = pyast.Expression(body=lam)
    pyast.fix_missing_locations(wrapped)
    code = compile(wrapped, "<math2osc_dsl>", "eval")
    env = build_eval_env(func_registry, sr=sr, dt=dt, extra=extra_env)
    env["__builtins__"] = {}
    return eval(code, env)


def compile_dsl(expr: str, *, sr: float = 48_000.0):
    dsl_ast = parse_dsl(expr)
    return build_callable(dsl_ast, sr=sr)
