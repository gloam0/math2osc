from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np

CONSTANT_NAMES = frozenset({"pi", "tau", "e", "sr", "dt"})
PARAM_SYMBOL = "x"
SUM_KEYWORD = "sum"
STEP_KEYWORD = "step"


@dataclass(frozen=True)
class FunctionSpec:
    name: str
    target: str
    arity: int
    impl: Callable


def _cot(x):
    return 1.0 / np.tan(x)


def _sec(x):
    return 1.0 / np.cos(x)


def _csc(x):
    return 1.0 / np.sin(x)


def _acot(x):
    return np.pi / 2.0 - np.arctan(x)


def _asec(x):
    return np.arccos(1.0 / x)


def _acsc(x):
    return np.arcsin(1.0 / x)


def _coth(x):
    return 1.0 / np.tanh(x)


def _sech(x):
    return 1.0 / np.cosh(x)


def _csch(x):
    return 1.0 / np.sinh(x)


def default_func_registry() -> Dict[str, FunctionSpec]:
    return {
        "sin": FunctionSpec("sin", "sin", 1, np.sin),
        "cos": FunctionSpec("cos", "cos", 1, np.cos),
        "tan": FunctionSpec("tan", "tan", 1, np.tan),
        "cot": FunctionSpec("cot", "cot", 1, _cot),
        "sec": FunctionSpec("sec", "sec", 1, _sec),
        "csc": FunctionSpec("csc", "csc", 1, _csc),
        "asin": FunctionSpec("asin", "asin", 1, np.arcsin),
        "acos": FunctionSpec("acos", "acos", 1, np.arccos),
        "atan": FunctionSpec("atan", "atan", 1, np.arctan),
        "acot": FunctionSpec("acot", "acot", 1, _acot),
        "asec": FunctionSpec("asec", "asec", 1, _asec),
        "acsc": FunctionSpec("acsc", "acsc", 1, _acsc),
        "sinh": FunctionSpec("sinh", "sinh", 1, np.sinh),
        "cosh": FunctionSpec("cosh", "cosh", 1, np.cosh),
        "tanh": FunctionSpec("tanh", "tanh", 1, np.tanh),
        "coth": FunctionSpec("coth", "coth", 1, _coth),
        "sech": FunctionSpec("sech", "sech", 1, _sech),
        "csch": FunctionSpec("csch", "csch", 1, _csch),
        "asinh": FunctionSpec("asinh", "asinh", 1, np.arcsinh),
        "acosh": FunctionSpec("acosh", "acosh", 1, np.arccosh),
        "atanh": FunctionSpec("atanh", "atanh", 1, np.arctanh),
    }


DEFAULT_FUNC_REGISTRY = default_func_registry()
