import unittest

import numpy as np
from sympy import sympify, Eq

from qctoolkit.expressions import Expression, ExpressionVariableMissingException, NonNumericEvaluation
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

        with self.assertRaises(NonNumericEvaluation):
            params['a'] = sympify('h')
            e.evaluate_numeric(**params)

    def test_evaluate_numpy(self):
        e = Expression('a * b + c')
        params = {
            'a': 2*np.ones(4),
            'b': 1.5*np.ones(4),
            'c': -7*np.ones(4)
        }
        np.testing.assert_equal((2 * 1.5 - 7) * np.ones(4), e.evaluate_numeric(**params))

    def test_evaluate_numeric_without_numpy(self):
        e = Expression('a * b + c', numpy_evaluation=False)

        params = {
            'a': 2,
            'b': 1.5,
            'c': -7
        }
        self.assertEqual(2 * 1.5 - 7, e.evaluate_numeric(**params))

        params = {
            'a': 2j,
            'b': 1.5,
            'c': -7
        }
        self.assertEqual(2j * 1.5 - 7, e.evaluate_numeric(**params))

        params = {
            'a': 2,
            'b': 6,
            'c': -7
        }
        self.assertEqual(2 * 6 - 7, e.evaluate_numeric(**params))

        params = {
            'a': 2,
            'b': sympify('k'),
            'c': -7
        }
        with self.assertRaises(NonNumericEvaluation):
            e.evaluate_numeric(**params)

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
        self.assertEqual("Expression('a    *    b')", repr(e))

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

    def assertExpressionEqual(self, lhs: Expression, rhs: Expression):
        self.assertTrue(bool(Eq(lhs.sympified_expression, rhs.sympified_expression)), '{} and {} are not equal'.format(lhs, rhs))

    def test_number_math(self):
        a = Expression('a')
        b = 3.3

        self.assertExpressionEqual(a + b, b + a)
        self.assertExpressionEqual(a - b, -(b - a))
        self.assertExpressionEqual(a * b, b * a)
        self.assertExpressionEqual(a / b, 1 / (b / a))

    def test_symbolic_math(self):
        a = Expression('a')
        b = Expression('b')

        self.assertExpressionEqual(a + b, b + a)
        self.assertExpressionEqual(a - b, -(b - a))
        self.assertExpressionEqual(a * b, b * a)
        self.assertExpressionEqual(a / b, 1 / (b / a))

    def test_sympy_math(self):
        a = Expression('a')
        b = sympify('b')

        self.assertExpressionEqual(a + b, b + a)
        self.assertExpressionEqual(a - b, -(b - a))
        self.assertExpressionEqual(a * b, b * a)
        self.assertExpressionEqual(a / b, 1 / (b / a))

    def test_get_most_simple_representation(self):
        cpl = Expression('1 + 1j').get_most_simple_representation()
        self.assertIsInstance(cpl, complex)
        self.assertEqual(cpl, 1 + 1j)

        integer = Expression('3').get_most_simple_representation()
        self.assertIsInstance(integer, int)
        self.assertEqual(integer, 3)

        flt = Expression('3.').get_most_simple_representation()
        self.assertIsInstance(flt, float)
        self.assertEqual(flt, 3.)

    def test_is_nan(self):
        self.assertTrue(Expression('nan').is_nan())
        self.assertTrue(Expression('0./0.').is_nan())

        self.assertFalse(Expression(456).is_nan())


class ExpressionExceptionTests(unittest.TestCase):
    def test_expression_variable_missing(self):
        variable = 's'
        expression = Expression('s*t')

        self.assertEqual(str(ExpressionVariableMissingException(variable, expression)),
                         "Could not evaluate <s*t>: A value for variable <s> is missing!")

    def test_non_numeric_evaluation(self):
        expression = Expression('a*b')
        call_arguments = dict()

        expected = "The result of evaluate_numeric is of type {} " \
                   "which is not a number".format(float)
        self.assertEqual(str(NonNumericEvaluation(expression, 1., call_arguments)), expected)

        expected = "The result of evaluate_numeric is of type {} " \
                   "which is not a number".format(np.zeros(1).dtype)
        self.assertEqual(str(NonNumericEvaluation(expression, np.zeros(1), call_arguments)), expected)
