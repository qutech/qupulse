import unittest

import numpy as np
from sympy import sympify

from qctoolkit.expressions import Expression, ExpressionVariableMissingException


class ExpressionTests(unittest.TestCase):

    def test_evaluate_numeric(self) -> None:
        e = Expression('a * b + c')
        params = {
            'a': 2,
            'b': 1.5,
            'c': -7
        }
        self.assertEqual(2 * 1.5 - 7, e.evaluate_numeric(**params))

    def test_evaluate_numpy(self):
        self
        e = Expression('a * b + c')
        params = {
            'a': 2*np.ones(4),
            'b': 1.5*np.ones(4),
            'c': -7*np.ones(4)
        }
        np.testing.assert_equal((2 * 1.5 - 7) * np.ones(4), e.evaluate_numeric(**params))

    def test_evaluate_symbolic(self):
        e = Expression('a * b + c')
        params = {
            'a': 'd',
            'c': -7
        }
        result = e.evaluate_symbolic(params)
        expected = sympify('d*b-7')
        self.assertEqual(result, expected)

    def test_variables(self) -> None:
        e = Expression('4 ** pi + x * foo')
        expected = sorted(['foo', 'x'])
        received = sorted(e.variables())
        self.assertEqual(expected, received)

    def test_evaluate_variable_missing(self) -> None:
        e = Expression('a * b + c')
        params = {
            'b': 1.5
        }
        with self.assertRaises(ExpressionVariableMissingException):
            e.evaluate_numeric(**params)
