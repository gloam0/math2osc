from dataclasses import dataclass
from typing import List, Optional, Tuple

Span = Tuple[int, int]

@dataclass(frozen=True)
class Node:
    span: Span

@dataclass(frozen=True)
class Number(Node):
    val: float

@dataclass(frozen=True)
class Constant(Node):
    name: str

@dataclass(frozen=True)
class X(Node):
    pass

@dataclass(frozen=True)
class Var(Node):
    name: str

@dataclass(frozen=True)
class UnaryOp(Node):
    node: Node

@dataclass(frozen=True)
class Neg(UnaryOp):
    pass

@dataclass(frozen=True)
class Pos(UnaryOp):
    pass

@dataclass(frozen=True)
class BinaryOp(Node):
    l: Node
    r: Node

@dataclass(frozen=True)
class Add(BinaryOp):
    pass

@dataclass(frozen=True)
class Sub(BinaryOp):
    pass

@dataclass(frozen=True)
class Mul(BinaryOp):
    pass

@dataclass(frozen=True)
class Div(BinaryOp):
    pass

@dataclass(frozen=True)
class Exp(BinaryOp):
    pass

@dataclass(frozen=True)
class FuncCall(Node):
    op: str
    args: List[Node]

@dataclass(frozen=True)
class Sum(Node):
    var: str
    lo: Node
    hi: Node
    step: Optional[Node]
    expr: Node
