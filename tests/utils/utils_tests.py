import unittest
from unittest import mock
from collections import OrderedDict

from qupulse.utils import checked_int_cast, replace_multiple


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
