import unittest

import numpy as np
from sympy import sympify, Eq

from qupulse.expressions import Expression, ExpressionVariableMissingException, NonNumericEvaluation, ExpressionScalar, ExpressionVector


class ExpressionTests(unittest.TestCase):
    def test_make(self):
        self.assertTrue(Expression.make('a') == 'a')
        self.assertTrue(Expression.make('a + b') == 'a + b')
        self.assertTrue(Expression.make(9) == 9)

        self.assertIsInstance(Expression.make([1, 'a']), ExpressionVector)

        self.assertIsInstance(ExpressionScalar.make('a'), ExpressionScalar)
        self.assertIsInstance(ExpressionVector.make(['a']), ExpressionVector)


class ExpressionVectorTests(unittest.TestCase):
    def test_evaluate_numeric(self) -> None:
        e = ExpressionVector(['a * b + c', 'a + d'])
        params = {
            'a': 2,
            'b': 1.5,
            'c': -7,
            'd': 9
        }
        np.testing.assert_equal(np.array([2 * 1.5 - 7, 2 + 9]),
                                e.evaluate_numeric(**params))

        with self.assertRaises(NonNumericEvaluation):
            params['a'] = sympify('h')
            e.evaluate_numeric(**params)

    def test_evaluate_numeric_2d(self) -> None:
        e = ExpressionVector([['a * b + c', 'a + d'], ['a', 3]])
        params = {
            'a': 2,
            'b': 1.5,
            'c': -7,
            'd': 9
        }
        np.testing.assert_equal(np.array([[2 * 1.5 - 7, 2 + 9], [2, 3]]),
                                e.evaluate_numeric(**params))

        with self.assertRaises(NonNumericEvaluation):
            params['a'] = sympify('h')
            e.evaluate_numeric(**params)

    def test_partial_evaluation(self):
        e = ExpressionVector(['a * b + c', 'a + d'])

        params = {
            'a': 2,
            'b': 1.5,
            'c': -7
        }

        expected = ExpressionVector([2 * 1.5 - 7, '2 + d'])
        evaluated = e.evaluate_symbolic(params)

        np.testing.assert_equal(evaluated.underlying_expression, expected.underlying_expression)

    def test_symbolic_evaluation(self):
        e = ExpressionVector([['a * b + c', 'a + d'], ['a', 3]])
        params = {
            'a': 2,
            'b': 1.5,
            'c': -7,
            'd': 9
        }

        expected = ExpressionVector([[2 * 1.5 - 7, 2 + 9], [2, 3]])
        evaluated = e.evaluate_symbolic(params)

        np.testing.assert_equal(evaluated.underlying_expression, expected.underlying_expression)

    def test_numeric_expression(self):
        numbers = np.linspace(1, 2, num=5)

        e = ExpressionVector(numbers)

        np.testing.assert_equal(e.underlying_expression, numbers)

    def test_eq(self):
        e1 = ExpressionVector([1, 2])
        e2 = ExpressionVector(['1', '2'])
        e3 = ExpressionVector(['1', 'a'])
        e4 = ExpressionVector([1, 'a'])
        e5 = ExpressionVector([1, 'a', 3])
        e6 = ExpressionVector([1, 1, '1'])
        e7 = ExpressionVector(['a'])

        self.assertEqual(e1, e2)
        self.assertEqual(e3, e4)
        self.assertNotEqual(e4, e5)

        self.assertEqual(e1, [1, 2])
        self.assertNotEqual(e6, 1)
        self.assertEqual(e7, ExpressionScalar('a'))


class ExpressionScalarTests(unittest.TestCase):
    def test_evaluate_numeric(self) -> None:
        e = ExpressionScalar('a * b + c')
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
        e = ExpressionScalar('a * b + c')
        params = {
            'a': 2*np.ones(4),
            'b': 1.5*np.ones(4),
            'c': -7*np.ones(4)
        }
        np.testing.assert_equal((2 * 1.5 - 7) * np.ones(4), e.evaluate_numeric(**params))

    def test_indexing(self):
        e = ExpressionScalar('a[i] * c')

        params = {
            'a': np.array([1, 2, 3]),
            'i': 1,
            'c': 2
        }

        self.assertEqual(e.evaluate_numeric(**params), 2 * 2)
        params['a'] = [1, 2, 3]
        self.assertEqual(e.evaluate_numeric(**params), 2 * 2)
        params['a'] = np.array([[1, 2, 3], [4, 5, 6]])
        np.testing.assert_equal(e.evaluate_numeric(**params), 2 * np.array([4, 5, 6]))

    def test_partial_evaluation(self) -> None:
        e = ExpressionScalar('a * c')
        params = {'c': 5.5}
        evaluated = e.evaluate_symbolic(params)
        expected = ExpressionScalar('a * 5.5')
        self.assertEqual(expected.underlying_expression, evaluated.underlying_expression)

    def test_partial_evaluation_vectorized(self) -> None:
        e = ExpressionScalar('a[i] * c')

        params = {
            'c': np.array([[1, 2], [3, 4]])
        }

        evaluated = e.evaluate_symbolic(params)
        expected = ExpressionVector([['a[i] * 1', 'a[i] * 2'], ['a[i] * 3', 'a[i] * 4']])

        np.testing.assert_equal(evaluated.underlying_expression, expected.underlying_expression)

    def test_evaluate_numeric_without_numpy(self):
        e = Expression('a * b + c')

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
        e = ExpressionScalar('a * b + c')
        params = {
            'a': 'd',
            'c': -7
        }
        result = e.evaluate_symbolic(params)
        expected = ExpressionScalar('d*b-7')
        self.assertEqual(result, expected)

    def test_variables(self) -> None:
        e = ExpressionScalar('4 ** pi + x * foo')
        expected = sorted(['foo', 'x'])
        received = sorted(e.variables)
        self.assertEqual(expected, received)

    def test_variables_indexed(self):
        e = ExpressionScalar('a[i] * c')
        expected = sorted(['a', 'i', 'c'])
        received = sorted(e.variables)
        self.assertEqual(expected, received)

    def test_evaluate_variable_missing(self) -> None:
        e = ExpressionScalar('a * b + c')
        params = {
            'b': 1.5
        }
        with self.assertRaises(ExpressionVariableMissingException):
            e.evaluate_numeric(**params)

    def test_repr(self):
        s = 'a    *    b'
        e = ExpressionScalar(s)
        self.assertEqual("Expression('a    *    b')", repr(e))

    def test_str(self):
        s = 'a    *    b'
        e = ExpressionScalar(s)
        self.assertEqual('a*b', str(e))

    def test_original_expression(self):
        s = 'a    *    b'
        self.assertEqual(ExpressionScalar(s).original_expression, s)

    def test_hash(self):
        expected = {ExpressionScalar(2), ExpressionScalar('a')}
        sequence = [ExpressionScalar(2), ExpressionScalar('a'), ExpressionScalar(2), ExpressionScalar('a')]
        self.assertEqual(expected, set(sequence))

    def test_undefined_comparison(self):
        valued = ExpressionScalar(2)
        unknown = ExpressionScalar('a')

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
        small = ExpressionScalar(2)
        large = ExpressionScalar(3)

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
        valued = ExpressionScalar(2)

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
        a = ExpressionScalar('a')
        b = 3.3

        self.assertExpressionEqual(a + b, b + a)
        self.assertExpressionEqual(a - b, -(b - a))
        self.assertExpressionEqual(a * b, b * a)
        self.assertExpressionEqual(a / b, 1 / (b / a))

    def test_symbolic_math(self):
        a = ExpressionScalar('a')
        b = ExpressionScalar('b')

        self.assertExpressionEqual(a + b, b + a)
        self.assertExpressionEqual(a - b, -(b - a))
        self.assertExpressionEqual(a * b, b * a)
        self.assertExpressionEqual(a / b, 1 / (b / a))

    def test_sympy_math(self):
        a = ExpressionScalar('a')
        b = sympify('b')

        self.assertExpressionEqual(a + b, b + a)
        self.assertExpressionEqual(a - b, -(b - a))
        self.assertExpressionEqual(a * b, b * a)
        self.assertExpressionEqual(a / b, 1 / (b / a))

    def test_is_nan(self):
        self.assertTrue(ExpressionScalar('nan').is_nan())
        self.assertTrue(ExpressionScalar('0./0.').is_nan())

        self.assertFalse(ExpressionScalar(456).is_nan())

    def test_special_function_numeric_evaluation(self):
        expr = Expression('erfc(t)')
        data = [-1., 0., 1.]
        expected = np.array([1.84270079, 1., 0.15729921])
        result = expr.evaluate_numeric(t=data)

        np.testing.assert_allclose(expected, result)


class ExpressionExceptionTests(unittest.TestCase):
    def test_expression_variable_missing(self):
        variable = 's'
        expression = ExpressionScalar('s*t')

        self.assertEqual(str(ExpressionVariableMissingException(variable, expression)),
                         "Could not evaluate <s*t>: A value for variable <s> is missing!")

    def test_non_numeric_evaluation(self):
        expression = ExpressionScalar('a*b')
        call_arguments = dict()

        expected = "The result of evaluate_numeric is of type {} " \
                   "which is not a number".format(float)
        self.assertEqual(str(NonNumericEvaluation(expression, 1., call_arguments)), expected)

        expected = "The result of evaluate_numeric is of type {} " \
                   "which is not a number".format(np.zeros(1).dtype)
        self.assertEqual(str(NonNumericEvaluation(expression, np.zeros(1), call_arguments)), expected)
