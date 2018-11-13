import sys
import unittest
import builtins
import contextlib
import importlib
import fractions
from unittest import mock

try:
    import gmpy2
except ImportError:
    gmpy2 = None

import qupulse.utils.types as qutypes


@contextlib.contextmanager
def mock_missing_module(module_name: str):
    exit_stack = contextlib.ExitStack()

    if module_name in sys.modules:
        # temporarily remove gmpy2 from the imported modules

        temp_modules = sys.modules.copy()
        del temp_modules[module_name]
        exit_stack.enter_context(mock.patch.dict(sys.modules, temp_modules))

    original_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == module_name:
            raise ImportError(name)
        else:
            return original_import(name, *args, **kwargs)

    exit_stack.enter_context(mock.patch('builtins.__import__', mock_import))

    with exit_stack:
        yield


class TestTimeType(unittest.TestCase):
    _fallback_qutypes = None

    @property
    def fallback_qutypes(self):
        if not self._fallback_qutypes:
            if gmpy2:
                with mock_missing_module('gmpy2'):
                    self._fallback_qutypes = importlib.reload(qutypes)

            else:
                self._fallback_qutypes = qutypes
        return self._fallback_qutypes

    def test_fraction_fallback(self):
        self.assertIs(fractions.Fraction, self.fallback_qutypes.TimeType)

    @unittest.skipIf(gmpy2 is None, "gmpy2 not available.")
    def test_default_time_from_float(self):
        # assert mocking did no permanent damage
        self.assertIs(gmpy2.mpq, qutypes.TimeType)

        self.assertEqual(qutypes.time_from_float(123/931), gmpy2.mpq(123, 931))

        self.assertEqual(qutypes.time_from_float(1000000/1000001, 1e-5), gmpy2.mpq(1))

    def test_fraction_time_from_float(self):
        self.assertEqual(self.fallback_qutypes.time_from_float(123 / 931),
                         fractions.Fraction(123, 931))

        self.assertEqual(self.fallback_qutypes.time_from_float(1000000 / 1000001, 1e-5),
                         fractions.Fraction(1))

