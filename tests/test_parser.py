import numpy as np
import pytest

from parser.ast_nodes import Sum
from parser import LoweringError, ParseError, build_callable, build_eval_env, compile_dsl, expand_sums, node_to_expanded_str, parse_dsl


class TestNodes:
    def test_number(self):
        f = compile_dsl("2.5")
        assert f(0.0) == 2.5
        assert f(123.0) == 2.5

    def test_x(self):
        f = compile_dsl("x")
        assert f(0.5) == 0.5

    def test_constant_pi(self):
        f = compile_dsl("pi")
        assert f(0.0) == float(np.pi)

    def test_unary(self):
        f = compile_dsl("-x")
        assert f(0.5) == -0.5

        g = compile_dsl("+x")
        assert g(0.5) == 0.5

    def test_binary_ops(self):
        f = compile_dsl("x + 2")
        assert f(0.5) == 2.5

        g = compile_dsl("x - 2")
        assert g(0.5) == -1.5

        h = compile_dsl("x * 2")
        assert h(0.5) == 1.0

        k = compile_dsl("x / 2")
        assert k(0.5) == 0.25

    def test_power_right_assoc(self):
        f = compile_dsl("2^3^2")
        assert f(0.0) == 512.0


class TestCorrectSolutions:
    def test_precedence(self):
        f = compile_dsl("1 + 2*3")
        assert f(0.0) == 7.0
        assert f(123.0) == 7.0

    def test_unary_vs_power(self):
        f = compile_dsl("-2^2")
        assert f(0.0) == -4.0

        g = compile_dsl("(-2)^2")
        assert g(0.0) == 4.0

    def test_negative_exponent(self):
        f = compile_dsl("2^-3")
        assert f(0.0) == 0.125

    def test_trig_unary(self):
        f = compile_dsl("sin(x)")
        assert np.allclose(f(0.5), np.sin(0.5))

        g = compile_dsl("sin(sin(x))")
        x = np.linspace(-1.0, 1.0, 31)
        assert np.allclose(g(x), np.sin(np.sin(x)))

    def test_constants_sr_dt(self):
        sr = 48_000.0
        f = compile_dsl("sr", sr=sr)
        assert f(0.0) == sr

        g = compile_dsl("dt", sr=sr)
        assert np.allclose(g(0.0), 1.0 / sr)


class TestVectorization:
    def test_vectorized_math(self):
        f = compile_dsl("sin(x) + cos(x)")
        x = np.linspace(0.0, 2.0, 17)
        assert np.allclose(f(x), np.sin(x) + np.cos(x))

class TestSum:
    def test_sum_is_explicit_ast_node_until_expanded(self):
        expr = "sum(k=1:3, k*x)"
        dsl_ast = parse_dsl(expr)
        assert isinstance(dsl_ast, Sum)
        expanded_ast = expand_sums(dsl_ast)
        assert not isinstance(expanded_ast, Sum)

    def test_sum_basic(self):
        f = compile_dsl("sum(k=1:3, k)")
        assert f(0.0) == 6.0
        assert f(123.0) == 6.0

    def test_sum_step(self):
        f = compile_dsl("sum(k=1:10, step=2, k)")
        assert f(0.0) == 25.0

    def test_sum_with_x(self):
        f = compile_dsl("sum(k=1:3, k*x)")
        x = np.linspace(0.0, 1.0, 5)
        assert np.allclose(f(x), 6.0 * x)

    def test_sum_trig(self):
        expr = "sum(k=1:5, step=2, sin(k*x)/k)"
        f = compile_dsl(expr)
        x = np.linspace(0.0, 1.0, 11)
        expected = sum(np.sin(k * x) / k for k in range(1, 6, 2))
        assert np.allclose(f(x), expected)

    def test_sum_expanded_str(self):
        expr = "sum(k=1:3, k*x)"
        dsl_ast = parse_dsl(expr)
        expanded = node_to_expanded_str(dsl_ast)
        x = np.linspace(0.0, 1.0, 7)
        env = build_eval_env()
        env["x"] = x
        env["__builtins__"] = {}
        expanded_val = eval(expanded, env)
        f = compile_dsl(expr)
        assert np.allclose(expanded_val, f(x))

    def test_sum_float_bounds(self):
        f = compile_dsl("sum(k=1.5:3, k)")
        assert f(0.0) == 4.0  # 1.5 + 2.5 = 4.0

    def test_sum_bounds_allow_functions(self):
        f = compile_dsl("sum(k=sin(0):1+sin(pi/2), k)")
        assert np.allclose(f(0.0), 3.0)

    def test_sum_bounds_allow_sr_dt(self):
        f = compile_dsl("sum(k=1:sr*dt*3, k)", sr=96_000.0)
        assert np.allclose(f(0.0), 6.0)

    def test_sum_many_terms_does_not_overflow_recursion(self):
        expr = "sum(k=1:400, sin(k*x)/(k^1.2))"
        dsl_ast = parse_dsl(expr)
        expanded = node_to_expanded_str(dsl_ast)
        assert "sin" in expanded
        f = compile_dsl(expr)
        x = np.linspace(0.0, 1.0, 16)
        y = f(x)
        assert np.all(np.isfinite(y))


class TestErrors:
    def test_parse_error_empty_args(self):
        with pytest.raises(ParseError):
            compile_dsl("sin()")

    def test_parse_error_incomplete(self):
        with pytest.raises(ParseError):
            compile_dsl("1 +")

    def test_lowering_error_wrong_arity(self):
        dsl_ast = parse_dsl("cos(1, 2)")
        with pytest.raises(LoweringError):
            build_callable(dsl_ast, sr=48_000.0)

    def test_lowering_error_unknown_variable(self):
        with pytest.raises(LoweringError):
            compile_dsl("k")

    def test_lowering_error_sum_bound_depends_on_x(self):
        with pytest.raises(LoweringError):
            compile_dsl("sum(k=1:x, k)")

    def test_parse_error_sum_var_cannot_be_x(self):
        with pytest.raises(ParseError):
            parse_dsl("sum(x=1:3, x)")

    def test_parse_error_sum_var_cannot_shadow_constant(self):
        with pytest.raises(ParseError):
            parse_dsl("sum(pi=1:3, pi)")

    def test_lowering_error_sum_bounds_must_be_finite(self):
        with pytest.raises(LoweringError):
            compile_dsl("sum(k=1:1e309, k)")
