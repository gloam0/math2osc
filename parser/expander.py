import ast as pyast
import math
from contextlib import contextmanager
from typing import Iterable, Mapping, Optional

from .ast_nodes import Add, Constant, Div, Exp, FuncCall, Mul, Neg, Node, Number, Pos, Sub, Sum, Var, X
from .errors import LoweringError
from .language import DEFAULT_FUNC_REGISTRY, FunctionSpec, PARAM_SYMBOL
from .lowerer import ASTLowerer
from .runtime_env import build_eval_env

_SUM_REL_TOL = 1e-12
_SUM_ABS_TOL = 1e-9
_UNSET = object()
_BALANCED_ADD_THRESHOLD = 64


def _sum_isclose(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=_SUM_REL_TOL, abs_tol=_SUM_ABS_TOL)


def _sum_indices(lo: float, hi: float, step: float, *, span) -> Iterable[float]:
    if not (math.isfinite(lo) and math.isfinite(hi) and math.isfinite(step)):
        raise LoweringError("sum bounds and step must be finite numbers", span=span)
    if step == 0:
        raise LoweringError("sum step cannot be 0", span=span)
    if step > 0:
        if lo > hi and not _sum_isclose(lo, hi):
            raise LoweringError("sum lower bound exceeds upper bound with positive step", span=span)
        curr = lo
        while curr < hi or _sum_isclose(curr, hi):
            yield curr
            curr += step
    else:
        if lo < hi and not _sum_isclose(lo, hi):
            raise LoweringError("sum lower bound below upper bound with negative step", span=span)
        curr = lo
        while curr > hi or _sum_isclose(curr, hi):
            yield curr
            curr += step


def _build_add_tree(terms: list[Node], *, span) -> Node:
    if not terms:
        return Number(span=span, val=0.0)
    if len(terms) <= _BALANCED_ADD_THRESHOLD:
        acc = terms[0]
        for term in terms[1:]:
            acc = Add(span=span, l=acc, r=term)
        return acc
    level = terms
    while len(level) > 1:
        next_level: list[Node] = []
        i = 0
        while i < len(level):
            left = level[i]
            if i + 1 < len(level):
                right = level[i + 1]
                next_level.append(Add(span=span, l=left, r=right))
                i += 2
            else:
                next_level.append(left)
                i += 1
        level = next_level
    return level[0]


class _SumExpander:
    def __init__(
        self,
        *,
        func_registry: Mapping[str, FunctionSpec],
        var_env: Optional[Mapping[str, float]] = None,
        sr: Optional[float] = None,
        dt: Optional[float] = None,
    ):
        self.func_registry = dict(func_registry)
        self.var_env = dict(var_env or {})
        self.sr = sr
        self.dt = dt

    @contextmanager
    def _bound_var(self, name: str, value: float):
        prev = self.var_env.get(name, _UNSET)
        self.var_env[name] = value
        try:
            yield
        finally:
            if prev is _UNSET:
                self.var_env.pop(name, None)
            else:
                self.var_env[name] = prev

    def _eval_numeric(self, node: Node, *, what: str) -> float:
        lowered = ASTLowerer(
            param_name=PARAM_SYMBOL,
            func_registry=self.func_registry,
            var_env=self.var_env,
        ).lower(node)
        if any(
            isinstance(sub_node, pyast.Name) and sub_node.id == PARAM_SYMBOL
            for sub_node in pyast.walk(lowered)
        ):
            raise LoweringError(f"{what} cannot depend on {PARAM_SYMBOL}", span=node.span)
        wrapped = pyast.Expression(body=lowered)
        pyast.fix_missing_locations(wrapped)
        env = build_eval_env(self.func_registry, sr=self.sr, dt=self.dt)
        env["__builtins__"] = {}
        try:
            value = eval(compile(wrapped, "<math2osc_sum_bound>", "eval"), env)
        except Exception as exc:
            raise LoweringError(f"{what} must be a constant numeric expression", span=node.span) from exc
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise LoweringError(f"{what} must evaluate to a number", span=node.span) from exc

    def expand(self, node: Node) -> Node:
        if isinstance(node, (Number, Constant, X)):
            return node
        if isinstance(node, Var):
            if node.name in self.var_env:
                return Number(span=node.span, val=float(self.var_env[node.name]))
            return node
        if isinstance(node, Neg):
            return Neg(span=node.span, node=self.expand(node.node))
        if isinstance(node, Pos):
            return Pos(span=node.span, node=self.expand(node.node))
        if isinstance(node, Add):
            return Add(span=node.span, l=self.expand(node.l), r=self.expand(node.r))
        if isinstance(node, Sub):
            return Sub(span=node.span, l=self.expand(node.l), r=self.expand(node.r))
        if isinstance(node, Mul):
            return Mul(span=node.span, l=self.expand(node.l), r=self.expand(node.r))
        if isinstance(node, Div):
            return Div(span=node.span, l=self.expand(node.l), r=self.expand(node.r))
        if isinstance(node, Exp):
            return Exp(span=node.span, l=self.expand(node.l), r=self.expand(node.r))
        if isinstance(node, FuncCall):
            return FuncCall(span=node.span, op=node.op, args=[self.expand(arg) for arg in node.args])
        if isinstance(node, Sum):
            lo_node = self.expand(node.lo)
            hi_node = self.expand(node.hi)
            step_node = self.expand(node.step) if node.step is not None else None
            lo = self._eval_numeric(lo_node, what="sum lower bound")
            hi = self._eval_numeric(hi_node, what="sum upper bound")
            step = self._eval_numeric(step_node, what="sum step") if step_node is not None else 1.0
            terms: list[Node] = []
            for k in _sum_indices(lo, hi, step, span=node.span):
                with self._bound_var(node.var, k):
                    terms.append(self.expand(node.expr))
            return _build_add_tree(terms, span=node.span)
        raise LoweringError(f"unhandled node type in sum expansion: {type(node).__name__}", span=node.span)


def expand_sums(
    node: Node,
    *,
    func_registry: Mapping[str, FunctionSpec] = DEFAULT_FUNC_REGISTRY,
    var_env: Optional[Mapping[str, float]] = None,
    sr: Optional[float] = None,
    dt: Optional[float] = None,
) -> Node:
    return _SumExpander(
        func_registry=func_registry,
        var_env=var_env,
        sr=sr,
        dt=dt,
    ).expand(node)
