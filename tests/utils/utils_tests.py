import unittest
from unittest import mock
from collections import OrderedDict

from qupulse.utils import checked_int_cast, replace_multiple, _fallback_pairwise, to_next_multiple


class PairWiseTest(unittest.TestCase):
    def test_fallback(self):
        self.assertEqual([(0, 1), (1, 2), (2, 3), (3, 4)], list(_fallback_pairwise(range(5))))
        self.assertEqual([], list(_fallback_pairwise(range(1))))
        self.assertEqual([], list(_fallback_pairwise(range(0))))


class CheckedIntCastTest(unittest.TestCase):
    def test_int_forwarding(self):
        my_int = 6
        self.assertIs(my_int, checked_int_cast(my_int))

    def test_no_int_detection(self):
        with self.assertRaises(ValueError):
            checked_int_cast(0.5)

        with self.assertRaises(ValueError):
            checked_int_cast(-0.5)

        with self.assertRaises(ValueError):
            checked_int_cast(123124.2)

        with self.assertRaises(ValueError):
            checked_int_cast(123124 + 1e-6, epsilon=1e-10)

    def test_float_cast(self):
        self.assertEqual(6, checked_int_cast(6+1e-11))

        self.assertEqual(-6, checked_int_cast(-6 + 1e-11))

    def test_variable_epsilon(self):
        self.assertEqual(6, checked_int_cast(6 + 1e-11))

        with self.assertRaises(ValueError):
            checked_int_cast(6 + 1e-11, epsilon=1e-15)


class IsCloseTest(unittest.TestCase):
    def test_isclose_fallback(self):
        import math
        import importlib
        import builtins
        import qupulse.utils as qutils

        def dummy_is_close():
            pass

        if hasattr(math, 'isclose'):
            dummy_is_close = math.isclose

        original_import = builtins.__import__

        def mock_import_missing_isclose(name, globals=None, locals=None, fromlist=(), level=0):
            if name == 'math' and 'isclose' in fromlist:
                raise ImportError(name)
            else:
                return original_import(name, globals=globals, locals=locals, fromlist=fromlist, level=level)

        def mock_import_exsiting_isclose(name, globals=None, locals=None, fromlist=(), level=0):
            if name == 'math' and 'isclose' in fromlist:
                if not hasattr(math, 'isclose'):
                    math.isclose = dummy_is_close
                return math
            else:
                return original_import(name, globals=globals, locals=locals, fromlist=fromlist, level=level)

        with mock.patch('builtins.__import__', mock_import_missing_isclose):
            reloaded_qutils = importlib.reload(qutils)
            self.assertIs(reloaded_qutils.isclose, reloaded_qutils._fallback_is_close)

        with mock.patch('builtins.__import__', mock_import_exsiting_isclose):
            reloaded_qutils = importlib.reload(qutils)
            self.assertIs(reloaded_qutils.isclose, math.isclose)

        if math.isclose is dummy_is_close:
            # cleanup
            delattr(math, 'isclose')


class ReplacementTests(unittest.TestCase):
    def test_replace_multiple(self):
        replacements = {'asd': 'dfg', 'dfg': '77', r'\*': '99'}

        text = r'it is asd and dfg that \*'
        expected = 'it is dfg and 77 that 99'
        result = replace_multiple(text, replacements)
        self.assertEqual(result, expected)

    def test_replace_multiple_overlap(self):
        replacement_list = [('asd', '1'), ('asdf', '2')]
        replacements = OrderedDict(replacement_list)
        result = replace_multiple('asdf', replacements)
        self.assertEqual(result, '1f')

        replacements = OrderedDict(reversed(replacement_list))
        result = replace_multiple('asdf', replacements)
        self.assertEqual(result, '2')


class ToNextMultipleTests(unittest.TestCase):
    def test_to_next_multiple(self):
        from qupulse.utils.types import TimeType
        from qupulse.expressions import ExpressionScalar
        precision_digits = 12
        
        duration = TimeType.from_float(47.1415926535)
        evaluated = to_next_multiple(sample_rate=TimeType.from_float(2.4),quantum=16)(duration)
        expected = ExpressionScalar('160/3')
        self.assertEqual(evaluated, expected)
        
        duration = TimeType.from_float(3.1415926535)
        evaluated = to_next_multiple(sample_rate=TimeType.from_float(2.4),quantum=16,min_quanta=13)(duration)
        expected = ExpressionScalar('260/3')
        self.assertEqual(evaluated, expected)
        
        duration = 6185240.0000001
        evaluated = to_next_multiple(sample_rate=1.0,quantum=16,min_quanta=13)(duration).evaluate_numeric()
        expected = 6185248
        self.assertAlmostEqual(evaluated, expected, precision_digits)
        
        duration = 63.99
        evaluated = to_next_multiple(sample_rate=1.0,quantum=16,min_quanta=4)(duration).evaluate_numeric()
        expected = 64
        self.assertAlmostEqual(evaluated, expected, precision_digits)
        
        duration = 64.01
        evaluated = to_next_multiple(sample_rate=1.0,quantum=16,min_quanta=4)(duration).evaluate_numeric()
        expected = 80
        self.assertAlmostEqual(evaluated, expected, precision_digits)
        
        duration = 0.
        evaluated = to_next_multiple(sample_rate=1.0,quantum=16,min_quanta=13)(duration).evaluate_numeric()
        expected = 0.
        self.assertAlmostEqual(evaluated, expected, precision_digits)
        
        duration = ExpressionScalar('abc')
        evaluated = to_next_multiple(sample_rate=1.0,quantum=16,min_quanta=13)(duration).evaluate_in_scope(dict(abc=0.))
        expected = 0.
        self.assertAlmostEqual(evaluated, expected, precision_digits)
        
        duration = ExpressionScalar('q')
        evaluated = to_next_multiple(sample_rate=ExpressionScalar('w'),quantum=16,min_quanta=1)(duration).evaluate_in_scope(
                        dict(q=3.14159,w=1.0))
        expected = 16.
        self.assertAlmostEqual(evaluated, expected, precision_digits)
        
        