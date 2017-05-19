import unittest

import numpy as np
from sympy import sympify

from qctoolkit.expressions import Expression, ExpressionVariableMissingException
from qctoolkit.serialization import Serializer

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
        expected = Expression('d*b-7')
        self.assertEqual(result, expected)

    def test_variables(self) -> None:
        e = Expression('4 ** pi + x * foo')
        expected = sorted(['foo', 'x'])
        received = sorted(e.variables)
        self.assertEqual(expected, received)

    def test_evaluate_variable_missing(self) -> None:
        e = Expression('a * b + c')
        params = {
            'b': 1.5
        }
        with self.assertRaises(ExpressionVariableMissingException):
            e.evaluate_numeric(**params)

    def test_repr(self):
        s = 'a    *    b'
        e = Expression(s)
        self.assertEqual('Expression(a    *    b)', repr(e))

    def test_str(self):
        s = 'a    *    b'
        e = Expression(s)
        self.assertEqual('a*b', str(e))

    def test_original_expression(self):
        s = 'a    *    b'
        self.assertEqual(Expression(s).original_expression, s)

    def test_undefined_comparison(self):
        valued = Expression(2)
        unknown = Expression('a')

        self.assertIsNone(unknown < 0)
        self.assertIsNone(unknown > 0)
        self.assertIsNone(unknown >= 0)
        self.assertIsNone(unknown <= 0)
        self.assertFalse(unknown == 0)

        self.assertIsNone(0 < unknown)
        self.assertIsNone(0 > unknown)
        self.assertIsNone(0 <= unknown)
        self.assertIsNone(0 >= unknown)
        self.assertFalse(0 == unknown)

        self.assertIsNone(unknown < valued)
        self.assertIsNone(unknown > valued)
        self.assertIsNone(unknown >= valued)
        self.assertIsNone(unknown <= valued)
        self.assertFalse(unknown == valued)

        valued, unknown = unknown, valued
        self.assertIsNone(unknown < valued)
        self.assertIsNone(unknown > valued)
        self.assertIsNone(unknown >= valued)
        self.assertIsNone(unknown <= valued)
        self.assertFalse(unknown == valued)
        valued, unknown = unknown, valued

        self.assertFalse(unknown == valued)

    def test_defined_comparison(self):
        small = Expression(2)
        large = Expression(3)

        self.assertIs(small < small, False)
        self.assertIs(small > small, False)
        self.assertIs(small <= small, True)
        self.assertIs(small >= small, True)
        self.assertIs(small == small, True)

        self.assertIs(small < large, True)
        self.assertIs(small > large, False)
        self.assertIs(small <= large, True)
        self.assertIs(small >= large, False)
        self.assertIs(small == large, False)

        self.assertIs(large < small, False)
        self.assertIs(large > small, True)
        self.assertIs(large <= small, False)
        self.assertIs(large >= small, True)
        self.assertIs(large == small, False)

    def test_number_comparison(self):
        valued = Expression(2)

        self.assertIs(valued < 3, True)
        self.assertIs(valued > 3, False)
        self.assertIs(valued <= 3, True)
        self.assertIs(valued >= 3, False)

        self.assertIs(valued == 3, False)
        self.assertIs(valued == 2, True)
        self.assertIs(3 == valued, False)
        self.assertIs(2 == valued, True)

        self.assertIs(3 < valued, False)
        self.assertIs(3 > valued, True)
        self.assertIs(3 <= valued, False)
        self.assertIs(3 >= valued, True)

