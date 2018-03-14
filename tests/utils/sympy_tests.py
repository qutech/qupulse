import unittest
import contextlib

import sympy
import numpy as np

from sympy.abc import a, b, c, d, e, f, k, l, m, n, i, j
from sympy import sin, Sum, IndexedBase

a_ = IndexedBase(a)
b_ = IndexedBase(b)

from qctoolkit.utils.sympy import sympify as qc_sympify, substitute_with_eval, recursive_substitution, Len

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


class SympifyTests(unittest.TestCase):
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

    def test_len_sympify(self):
        if type(self) is SympifyTests:
            expected_exception = self.assertRaises(AssertionError, msg="sympy.sympify does not know len")
        else:
            expected_exception = contextlib.suppress()

        with expected_exception:
            for s, expected in len_sympify:
                result = self.sympify(s)
                self.assertEqual(result, expected)

    def test_index_sympify(self):
        if type(self) is SympifyTests:
            # This should fail if sympy start supporting
            expected_exception = self.assertRaises(TypeError, msg="sympy.sympify does not support indexing")
        else:
            expected_exception = contextlib.suppress()

        with expected_exception:
            for s, expected in index_sympify:
                result = self.sympify(s)
                self.assertEqual(result, expected)


class SympifyWrapperTests(SympifyTests):
    def sympify(self, expression):
        return qc_sympify(expression)


class SubstitutionTests(unittest.TestCase):
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
