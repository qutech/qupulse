import unittest

from qctoolkit.expressions import Expression, ExpressionVariableMissingException


class ExpressionTests(unittest.TestCase):

    def test_evaluate(self) -> None:
        e = Expression('a * b + c')
        params = {
            'a': 2,
            'b': 1.5,
            'c': -7
        }
        self.assertEqual(2*1.5 - 7, e.evaluate(**params))

    def test_variables(self) -> None:
        e = Expression('4 ** PI + x * foo')
        self.assertEqual(sorted(['foo','x']), sorted(e.variables()))

    def test_evaluate_variable_missing(self) -> None:
        e = Expression('a * b + c')
        params = {
            'b': 1.5
        }
        with self.assertRaises(ExpressionVariableMissingException):
            e.evaluate(**params)