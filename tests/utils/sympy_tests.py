import unittest
import contextlib
import math

from typing import Union

import sympy
import numpy as np

from sympy.abc import a, b, c, d, e, f, k, l, m, n, i, j
from sympy import sin, Sum, IndexedBase

a_ = IndexedBase(a)
b_ = IndexedBase(b)
foo_bar = sympy.Symbol('foo____bar')
foo_bar_ = IndexedBase('foo____bar')
scope_n = sympy.Symbol('scope____n')


from qupulse.utils.sympy import sympify as qc_sympify, substitute_with_eval, recursive_substitution, Len,\
    evaluate_lambdified, evaluate_compiled, get_most_simple_representation, get_variables, get_free_symbols,\
    almost_equal, substitute


################################################### SUBSTITUTION #######################################################
simple_substitution_cases = [
    (a*b, {'a': c}, b*c),
    (a*b, {'a': b, 'b': a}, a*b),
    (a*b, {'a': 1, 'b': 2}, 2),
    (foo_bar*b, {'foo.bar': c}, b*c),
    (foo_bar*b, {'b': scope_n}, scope_n*foo_bar),
    (foo_bar*b, {'b': foo_bar, 'foo.bar': b}, foo_bar*b),
    (foo_bar*scope_n, {'foo.bar': scope_n, 'scope.n': foo_bar}, foo_bar*scope_n),
    (foo_bar*b, {'foo.bar': 5, 'b': 0.2}, 0.2*5)
]

elem_func_substitution_cases = [
    (a*b + sin(c), {'a': b, 'c': sympy.pi/2}, b**2 + 1),
    (a*scope_n + sin(foo_bar), {'a': scope_n, 'foo.bar': sympy.pi/2}, scope_n**2 + 1)
]

sum_substitution_cases = [
    (a*b + Sum(c * k, (k, 0, n)), {'a': b, 'b': 2, 'k': 1, 'n': 2}, b*2 + c*(1 + 2)),
    (a*foo_bar + Sum(c * scope_n, (scope_n, 0, n)), {'a': foo_bar, 'foo.bar': 2, 'scope.n': 1, 'n': 2}, foo_bar*2 + c*(1 + 2)),
]

indexed_substitution_cases = [
    (a_[i] * b, {'b': 3}, a_[i] * 3),
    (a_[i] * b, {'a': sympy.Array([1, 2, 3])}, sympy.Array([1, 2, 3])[i] * b),
    (sympy.Array([1, 2, 3])[i] * b, {'i': 1}, 2 * b),
    (foo_bar_[i]*scope_n, {'scope.n': 3}, foo_bar_[i]*3),
    (foo_bar_[i]*scope_n, {'foo.bar': sympy.Array([1, 2, 3])}, sympy.Array([1, 2, 3])[i]*scope_n),
    (sympy.Array([1, 2, 3])[foo_bar]*b, {'foo.bar': 1}, 2*b),
]

vector_valued_cases = [
    (a*b, {'a': sympy.Array([1, 2, 3])}, sympy.Array([1, 2, 3])*b),
    (a*b, {'a': sympy.Array([1, 2, 3]), 'b': sympy.Array([4, 5, 6])}, sympy.Array([4, 10, 18])),
    (foo_bar*b, {'foo.bar': sympy.Array([1, 2, 3])}, sympy.Array([1, 2, 3])*b),
    (foo_bar*b, {'foo.bar': sympy.Array([1, 2, 3]), 'b': sympy.Array([4, 5, 6])}, sympy.Array([4, 10, 18])),
    (foo_bar*scope_n, {'foo.bar': sympy.Array([1, 2, 3]), 'scope.n': sympy.Array([4, 5, 6])}, sympy.Array([4, 10, 18])),
]

full_featured_cases = [
    (Sum(a_[i], (i, 0, Len(a) - 1)), {'a': sympy.Array([1, 2, 3])}, 6),
    (Sum(foo_bar_[i], (i, 0, Len(foo_bar) - 1)), {'foo.bar': sympy.Array([1, 2, 3])}, 6),
]


##################################################### SYMPIFY ##########################################################
simple_sympify = [
    ('a*b', a*b),
    ('a*6', a*6),
    ('sin(a)', sin(a)),
    ('foo.bar*6', foo_bar*6),
    ('sin(foo.bar)', sin(foo_bar))
]

complex_sympify = [
    ('Sum(a, (i, 0, n))', Sum(a, (i, 0, n))),
    ('Sum(foo.bar, (i, 0, scope.n))', Sum(foo_bar, (i, 0, scope_n)))
]

len_sympify = [
    ('len(a)', Len(a)),
    ('Len(a)', Len(a))
]

index_sympify = [
    ('a[i]', a_[i]),
    ('foo.bar[i]', foo_bar_[i])
]


#################################################### EVALUATION ########################################################
eval_simple = [
    (a*b, {'a': 2, 'b': 3}, 6),
    (a*b, {'a': 2, 'b': np.float32(3.5)}, 2*np.float32(3.5)),
    (a+b, {'a': 3.4, 'b': 76.7}, 3.4+76.7),
    (foo_bar+scope_n, {'foo.bar': 1.2, 'scope.n': 3.3}, 1.2+3.3),
    (foo_bar*scope_n, {'foo.bar': 1.2, 'scope.n': np.float32(3.3)}, 1.2*np.float32(3.3)),
    (foo_bar*scope_n, {'foo.bar': 1.2, 'scope.n': 3.3}, 1.2*3.3)
]

eval_many_arguments = [
    (sum(sympy.symbols(list('a_' + str(i) for i in range(300)))), {'a_' + str(i): 1 for i in range(300)}, 300),
    (sum(sympy.symbols(list('scope____a_' + str(i) for i in range(300)))), {'scope.a_' + str(i): 1 for i in range(300)}, 300)
]

eval_simple_functions = [
    (a*sin(b), {'a': 3.5, 'b': 1.2}, 3.5*math.sin(1.2)),
    (a*sin(foo_bar), {'a': 3.5, 'foo.bar': 1.2}, 3.5*math.sin(1.2)),
]

eval_array_values = [
    (a * b, {'a': 2, 'b': np.array([3])}, np.array([6])),
    (a * b, {'a': 2, 'b': np.array([3, 4, 5])}, np.array([6, 8, 10])),
    (a * b, {'a': np.array([2, 3]), 'b': np.array([100, 200])}, np.array([200, 600])),
    (a * foo_bar, {'a': 2, 'foo.bar': np.array([3])}, np.array([6])),
    (a * foo_bar, {'a': 2, 'foo.bar': np.array([3, 4, 5])}, np.array([6, 8, 10])),
    (a * foo_bar, {'a': np.array([2, 3]), 'foo.bar': np.array([100, 200])}, np.array([200, 600])),
]

eval_sum = [
    (Sum(a_[i], (i, 0, Len(a) - 1)), {'a': np.array([1, 2, 3])}, 6),
    (Sum(foo_bar_[i], (i, 0, Len(foo_bar) - 1)), {'foo.bar': np.array([1, 2, 3])}, 6),
]

eval_array_expression = [
    (np.array([a*c, b*c]), {'a': 2, 'b': 3, 'c': 4}, np.array([8, 12])),
    (np.array([a*foo_bar, scope_n*foo_bar]), {'a': 2, 'scope.n': 3, 'foo.bar': 4}, np.array([8, 12]))
]


class TestCase(unittest.TestCase):
    def assertRaises(self, expected_exception, *args, **kwargs):
        if expected_exception is None:
            return contextlib.suppress()
        else:
            return super().assertRaises(expected_exception, *args, **kwargs)


class SympifyTests(TestCase):

    def sympify(self, expression) -> sympy.Expr:
        return qc_sympify(expression)

    def assertEqual1(self, first, second, msg=None):
        if sympy.Eq(first, second):
            return
        raise self.failureException(msg=msg)

    def test_simple_sympify(self) -> None:
        for s, expected in simple_sympify:
            result = self.sympify(s)
            self.assertEqual(expected, result)

    def test_complex_sympify(self) -> None:
        for s, expected in complex_sympify:
            result = self.sympify(s)
            self.assertEqual(expected, result)

    def test_len_sympify(self) -> None:
        for s, expected in len_sympify:
            result = self.sympify(s)
            self.assertEqual(expected, result)

    def test_index_sympify(self) -> None:
        for s, expected in index_sympify:
            result = self.sympify(s)
            self.assertEqual(expected, result)


class SubstitutionTests(TestCase):
    def substitute(self, expression: sympy.Expr, substitutions: dict):
        for key, value in substitutions.items():
            if not isinstance(value, sympy.Expr):
                substitutions[key] = qc_sympify(value)
        return substitute(expression, substitutions, simultaneous=True).doit()

    def test_simple_substitution_cases(self):
        for expr, subs, expected in simple_substitution_cases:
            result = self.substitute(expr, subs)
            self.assertEqual(expected, result, msg=str((expr, subs, expected)))

    def test_elem_func_substitution_cases(self):
        for expr, subs, expected in elem_func_substitution_cases:
            result = self.substitute(expr, subs)
            self.assertEqual(expected, result)

    def test_sum_substitution_cases(self):
        for expr, subs, expected in sum_substitution_cases:
            result = self.substitute(expr, subs)
            self.assertEqual(expected, result)

    def test_indexed_substitution_cases(self):
        if type(self) is SubstitutionTests:
            raise unittest.SkipTest('sympy.Expr.subs does not handle simultaneous substitutions of indexed entities.')

        for expr, subs, expected in indexed_substitution_cases:
            result = self.substitute(expr, subs)
            self.assertEqual(expected, result)

    def test_vector_valued_cases(self):
        if type(self) is SubstitutionTests:
            raise unittest.SkipTest('sympy.Expr.subs does not handle simultaneous substitutions of indexed entities.')

        for expr, subs, expected in vector_valued_cases:
            result = self.substitute(expr, subs)
            self.assertEqual(expected, result, msg="test: {}".format((expr, subs, expected)))

    def test_full_featured_cases(self):
        if type(self) is SubstitutionTests:
            raise unittest.SkipTest('sympy.Expr.subs does not handle simultaneous substitutions of indexed entities.')

        for expr, subs, expected in full_featured_cases:
            result = self.substitute(expr, subs)
            self.assertEqual(expected, result)


class SubstituteWithEvalTests(SubstitutionTests):
    def substitute(self, expression: sympy.Expr, substitutions: dict):
        return substitute_with_eval(expression, substitutions)

    @unittest.expectedFailure
    def test_sum_substitution_cases(self):
        super().test_sum_substitution_cases()

    @unittest.expectedFailure
    def test_full_featured_cases(self):
        super().test_full_featured_cases()


class GetFreeSymbolsTests(TestCase):
    def assert_symbol_sets_equal(self, expected, actual):
        self.assertEqual(len(expected), len(actual))
        self.assertEqual(set(expected), set(actual))

    def test_get_free_symbols(self):
        expr = a * b / 5
        self.assert_symbol_sets_equal([a, b], get_free_symbols(expr))

    def test_get_free_symbols_indexed(self):
        expr = a_[i] * IndexedBase(a*b)[j]
        self.assert_symbol_sets_equal({a, b, i, j}, set(get_free_symbols(expr)))

    def test_get_variables(self):
        expr = a * b / 5
        self.assertEqual({'a', 'b'}, set(get_variables(expr)))

    def test_get_variables_indexed(self):
        expr = a_[i] * IndexedBase(a*b)[j]
        self.assertEqual({'a', 'b', 'i', 'j'}, set(get_variables(expr)))


class EvaluationTests(TestCase):
    def evaluate(self, expression: Union[sympy.Expr, np.ndarray], parameters):
        if isinstance(expression, np.ndarray):
            variables = set.union(*map(set, map(get_variables, expression.flat)))
        else:
            variables = get_variables(expression)
        return evaluate_lambdified(expression, variables=list(variables), parameters=parameters, lambdified=None)[0]

    def test_eval_simple(self):
        for expr, parameters, expected in eval_simple:
            result = self.evaluate(expr, parameters)
            self.assertEqual(expected, result)

    def test_eval_many_arguments(self, expected_exception=SyntaxError):
        with self.assertRaises(expected_exception):
            for expr, parameters, expected in eval_many_arguments:
                result = self.evaluate(expr, parameters)
                self.assertEqual(expected, result)

    def test_eval_simple_functions(self):
        for expr, parameters, expected in eval_simple_functions:
            result = self.evaluate(expr, parameters)
            self.assertEqual(expected, result)

    def test_eval_array_values(self):
        for expr, parameters, expected in eval_array_values:
            result = self.evaluate(expr, parameters)
            np.testing.assert_equal(expected, result)

    def test_eval_sum(self):
        for expr, parameters, expected in eval_sum:
            result = self.evaluate(expr, parameters)
            self.assertEqual(expected, result)

    def test_eval_array_expression(self):
        for expr, parameters, expected in eval_array_expression:
            result = self.evaluate(expr, parameters)
            np.testing.assert_equal(expected, result)


class CompiledEvaluationTest(EvaluationTests):
    def evaluate(self, expression: Union[sympy.Expr, np.ndarray], parameters):
        if isinstance(expression, np.ndarray):
            return self.evaluate(sympy.Array(expression), parameters)

        result, _ = evaluate_compiled(expression, parameters, compiled=None)

        if isinstance(result, (list, tuple)):
            return np.array(result)
        else:
            return result

    def test_eval_many_arguments(self):
        super().test_eval_many_arguments(None)


class RepresentationTest(unittest.TestCase):
    def test_get_most_simple_representation(self):
        cpl = get_most_simple_representation(qc_sympify('1 + 1j'))
        self.assertIsInstance(cpl, str)
        self.assertTrue(bool(sympy.Eq(sympy.sympify(cpl), 1 + 1j)))

        integer = get_most_simple_representation(qc_sympify('3'))
        self.assertIsInstance(integer, int)
        self.assertEqual(integer, 3)

        flt = get_most_simple_representation(qc_sympify('3.1'))
        self.assertIsInstance(flt, float)
        self.assertEqual(flt, 3.1)

        st = get_most_simple_representation(qc_sympify('a + b'))
        self.assertIsInstance(st, str)
        self.assertEqual(st, 'a + b')

        sym = get_most_simple_representation(qc_sympify('b + foo.bar.test'))
        self.assertIsInstance(sym, str)
        self.assertEqual('b + foo.bar.test', sym)


class AlmostEqualTests(unittest.TestCase):
    def test_almost_equal(self):
        self.assertTrue(almost_equal(sympy.sin(a) * 0.5, sympy.sin(a) / 2))
        self.assertIsNone(almost_equal(sympy.sin(a) * 0.5, sympy.sin(b) / 2))
        self.assertFalse(almost_equal(sympy.sin(a), sympy.sin(a) + 1e-14))

        self.assertTrue(almost_equal(sympy.sin(a), sympy.sin(a) + 1e-14, epsilon=1e-13))


class NamespaceTests(unittest.TestCase):

    def test_sympify_dot_namespace_notations(self) -> None:
        expr = qc_sympify("qubit.a + qubit.spec2.a * 1.3")
        expected = sympy.Add(sympy.Symbol('qubit____a'), sympy.Mul(sympy.Symbol('qubit____spec2____a'), sympy.RealNumber(1.3)))
        self.assertEqual(expected, expr)

    def test_evaluate_lambdified_dot_namespace_notation(self) -> None:
        res = evaluate_lambdified("qubit.a + qubit.spec2.a * 1.3", ["qubit.a", "qubit.spec2.a"], {"qubit.a": 2.1, "qubit.spec2.a": .1}, lambdified=None)
        self.assertEqual(2.23, res[0])

    def test_evaluate_compiled_dot_namespace_notation(self) -> None:
        res = evaluate_compiled("qubit.a + qubit.spec2.a * 1.3", {"qubit.a": 2.1, "qubit.spec2.a": .1})
        self.assertEqual(2.23, res[0])
