import unittest
import contextlib
import math
import sys

from typing import Union

import sympy
import numpy as np

from sympy.abc import a, b, c, d, e, f, k, l, m, n, i, j
from sympy import sin, Sum, IndexedBase

a_ = IndexedBase(a)
b_ = IndexedBase(b)

from qupulse.utils.sympy import sympify as qc_sympify, substitute_with_eval, recursive_substitution, Len,\
    evaluate_lambdified, evaluate_compiled, get_most_simple_representation, get_variables, get_free_symbols,\
    almost_equal


################################################### SUBSTITUTION #######################################################
simple_substitution_cases = [
    (a*b, {'a': c}, b*c),
    (a*b, {'a': b, 'b': a}, a*b),
    (a*b, {'a': 1, 'b': 2}, 2),
]

elem_func_substitution_cases = [
    (a*b + sin(c), {'a': b, 'c': sympy.pi/2}, b**2 + 1),
]

sum_substitution_cases = [
    (a*b + Sum(c * k, (k, 0, n)), {'a': b, 'b': 2, 'k': 1, 'n': 2}, b*2 + c*(1 + 2)),
]

indexed_substitution_cases = [
    (a_[i]*b, {'b': 3}, a_[i]*3),
    (a_[i]*b, {'a': sympy.Array([1, 2, 3])}, sympy.Array([1, 2, 3])[i]*b),
    (sympy.Array([1, 2, 3])[i]*b, {'i': 1}, 2*b)
]

vector_valued_cases = [
    (a*b, {'a': sympy.Array([1, 2, 3])}, sympy.Array([1, 2, 3])*b),
    (a*b, {'a': sympy.Array([1, 2, 3]), 'b': sympy.Array([4, 5, 6])}, sympy.Array([4, 10, 18])),
]

full_featured_cases = [
    (Sum(a_[i], (i, 0, Len(a) - 1)), {'a': sympy.Array([1, 2, 3])}, 6),
]


##################################################### SYMPIFY ##########################################################
simple_sympify = [
    ('a*b', a*b),
    ('a*6', a*6),
    ('sin(a)', sin(a))
]

complex_sympify = [
    ('Sum(a, (i, 0, n))', Sum(a, (i, 0, n))),
]

len_sympify = [
    ('len(a)', Len(a)),
    ('Len(a)', Len(a))
]

index_sympify = [
    ('a[i]', a_[i])
]


#################################################### EVALUATION ########################################################
eval_simple = [
    (a*b, {'a': 2, 'b': 3}, 6),
    (a*b, {'a': 2, 'b': np.float32(3.5)}, 2*np.float32(3.5)),
    (a+b, {'a': 3.4, 'b': 76.7}, 3.4+76.7)
]

eval_many_arguments = [
    (sum(sympy.symbols(list('a_' + str(i) for i in range(300)))), {'a_' + str(i): 1 for i in range(300)}, 300)
]

eval_simple_functions = [
    (a*sin(b), {'a': 3.5, 'b': 1.2}, 3.5*math.sin(1.2))
]

eval_array_values = [
    (a*b, {'a': 2, 'b': np.array([3])}, np.array([6])),
    (a*b, {'a': 2, 'b': np.array([3, 4, 5])}, np.array([6, 8, 10])),
    (a*b, {'a': np.array([2, 3]), 'b': np.array([100, 200])}, np.array([200, 600]))
]

eval_sum = [
    (Sum(a_[i], (i, 0, Len(a) - 1)), {'a': np.array([1, 2, 3])}, 6),
]

eval_array_expression = [
    (np.array([a*c, b*c]), {'a': 2, 'b': 3, 'c': 4}, np.array([8, 12]))
]


class TestCase(unittest.TestCase):
    def assertRaises(self, expected_exception, *args, **kwargs):
        if expected_exception is None:
            return contextlib.suppress()
        else:
            return super().assertRaises(expected_exception, *args, **kwargs)


class SympifyTests(TestCase):
    def sympify(self, expression):
        return sympy.sympify(expression)

    def assertEqual1(self, first, second, msg=None):
        if sympy.Eq(first, second):
            return
        raise self.failureException(msg=msg)

    def test_simple_sympify(self):
        for s, expected in simple_sympify:
            result = self.sympify(s)
            self.assertEqual(result, expected)

    def test_complex_sympify(self):
        for s, expected in complex_sympify:
            result = self.sympify(s)
            self.assertEqual(result, expected)

    def test_len_sympify(self, expected_exception=AssertionError, msg="sympy.sympify does not know len"):
        with self.assertRaises(expected_exception=expected_exception, msg=msg):
            for s, expected in len_sympify:
                result = self.sympify(s)
                self.assertEqual(result, expected)

    def test_index_sympify(self, expected_exception=TypeError):
        with self.assertRaises(expected_exception=expected_exception):
            for s, expected in index_sympify:
                result = self.sympify(s)
                self.assertEqual(result, expected)


class SympifyWrapperTests(SympifyTests):
    def sympify(self, expression):
        return qc_sympify(expression)

    def test_len_sympify(self):
        super().test_len_sympify(None)

    def test_index_sympify(self):
        super().test_index_sympify(None)


class SubstitutionTests(TestCase):
    def substitute(self, expression: sympy.Expr, substitutions: dict):
        for key, value in substitutions.items():
            if not isinstance(value, sympy.Expr):
                substitutions[key] = sympy.sympify(value)
        return expression.subs(substitutions, simultaneous=True).doit()

    def test_simple_substitution_cases(self):
        for expr, subs, expected in simple_substitution_cases:
            result = self.substitute(expr, subs)
            self.assertEqual(result, expected)

    def test_elem_func_substitution_cases(self):
        for expr, subs, expected in elem_func_substitution_cases:
            result = self.substitute(expr, subs)
            self.assertEqual(result, expected)

    def test_sum_substitution_cases(self):
        for expr, subs, expected in sum_substitution_cases:
            result = self.substitute(expr, subs)
            self.assertEqual(result, expected)

    def test_indexed_substitution_cases(self):
        if type(self) is SubstitutionTests:
            raise unittest.SkipTest('sympy.Expr.subs does not handle simultaneous substitutions of indexed entities.')

        for expr, subs, expected in indexed_substitution_cases:
            result = self.substitute(expr, subs)
            self.assertEqual(result, expected)

    def test_vector_valued_cases(self):
        if type(self) is SubstitutionTests:
            raise unittest.SkipTest('sympy.Expr.subs does not handle simultaneous substitutions of indexed entities.')

        for expr, subs, expected in vector_valued_cases:
            result = self.substitute(expr, subs)
            self.assertEqual(result, expected)

    def test_full_featured_cases(self):
        if type(self) is SubstitutionTests:
            raise unittest.SkipTest('sympy.Expr.subs does not handle simultaneous substitutions of indexed entities.')

        for expr, subs, expected in full_featured_cases:
            result = self.substitute(expr, subs)
            self.assertEqual(result, expected)


class SubstituteWithEvalTests(SubstitutionTests):
    def substitute(self, expression: sympy.Expr, substitutions: dict):
        return substitute_with_eval(expression, substitutions)

    @unittest.expectedFailure
    def test_sum_substitution_cases(self):
        super().test_sum_substitution_cases()

    @unittest.expectedFailure
    def test_full_featured_cases(self):
        super().test_full_featured_cases()


class RecursiveSubstitutionTests(SubstitutionTests):
    def substitute(self, expression: sympy.Expr, substitutions: dict):
        return recursive_substitution(expression, substitutions).doit()


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


class EvaluationTestsBase:

    def test_eval_simple(self):
        for expr, parameters, expected in eval_simple:
            result = self.evaluate(expr, parameters)
            self.assertEqual(result, expected)

    def test_eval_many_arguments(self):
        for expr, parameters, expected in eval_many_arguments:
            result = self.evaluate(expr, parameters)
            self.assertEqual(result, expected)

    def test_eval_simple_functions(self):
        for expr, parameters, expected in eval_simple_functions:
            result = self.evaluate(expr, parameters)
            self.assertEqual(result, expected)

    def test_eval_array_values(self):
        for expr, parameters, expected in eval_array_values:
            result = self.evaluate(expr, parameters)
            np.testing.assert_equal(result, expected)

    def test_eval_sum(self):
        for expr, parameters, expected in eval_sum:
            result = self.evaluate(expr, parameters)
            self.assertEqual(result, expected)

    def test_eval_array_expression(self):
        for expr, parameters, expected in eval_array_expression:
            result = self.evaluate(expr, parameters)
            np.testing.assert_equal(result, expected)


class LamdifiedEvaluationTest(EvaluationTestsBase, unittest.TestCase):

    def evaluate(self, expression: Union[sympy.Expr, np.ndarray], parameters):
        if isinstance(expression, np.ndarray):
            variables = set.union(*map(set, map(get_variables, expression.flat)))
        else:
            variables = get_variables(expression)
        return evaluate_lambdified(expression, variables=list(variables), parameters=parameters, lambdified=None)[0]

    @unittest.skipIf(sys.version_info[0] == 3 and sys.version_info[1] < 7, "causes syntax error for python < 3.7")
    def test_eval_many_arguments(self):
        super().test_eval_many_arguments()


class CompiledEvaluationTest(EvaluationTestsBase, unittest.TestCase):

    def evaluate(self, expression: Union[sympy.Expr, np.ndarray], parameters):
        if isinstance(expression, np.ndarray):
            return self.evaluate(sympy.Array(expression), parameters)

        result, _ = evaluate_compiled(expression, parameters, compiled=None)

        if isinstance(result, (list, tuple)):
            return np.array(result)
        else:
            return result

    def test_eval_many_arguments(self):
        super().test_eval_many_arguments()


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


class AlmostEqualTests(unittest.TestCase):
    def test_almost_equal(self):
        self.assertTrue(almost_equal(sympy.sin(a) * 0.5, sympy.sin(a) / 2))
        self.assertIsNone(almost_equal(sympy.sin(a) * 0.5, sympy.sin(b) / 2))
        self.assertFalse(almost_equal(sympy.sin(a), sympy.sin(a) + 1e-14))

        self.assertTrue(almost_equal(sympy.sin(a), sympy.sin(a) + 1e-14, epsilon=1e-13))
