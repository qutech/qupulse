import unittest

from qctoolkit.expressions import Expression


class ExpressionTests(unittest.TestCase):

    def test_evaluate(self) -> None:
        e = Expression('a * b + c')
        params = {
            'a': 2,
            'b': 1.5,
            'c': -7
        }
        self.assertEqual(2*1.5 - 7, e.evaluate(**params))