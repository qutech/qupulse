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

import numpy as np

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
        self.assertIs(fractions.Fraction, self.fallback_qutypes.TimeType._InternalRepresentation)

    def test_fraction_time_from_float_exact(self):
        self.assertEqual(self.fallback_qutypes.time_from_float(123 / 931, 0),
                         fractions.Fraction(123 / 931))

    def test_fraction_time_from_float_with_precision(self):
        self.assertEqual(self.fallback_qutypes.time_from_float(1000000 / 1000001, 1e-5),
                         fractions.Fraction(1))

    def test_fraction_time_from_fraction(self):
        t = qutypes.TimeType.from_fraction(43, 12)
        self.assertIsInstance(t, qutypes.TimeType)
        self.assertEqual(t, fractions.Fraction(43, 12))

    def test_from_float_no_extra_args(self):
        # test that float(from_float(x)) == x
        base_floats = [4/5, 1, 1000, 0, np.pi, 1.23456789**99, 1e-100, 2**53]
        n_steps = 10**2

        def float_generator():
            for f in base_floats:
                for _ in range(n_steps):
                    yield f
                    f = np.nextafter(f, float('inf'))

            for f in base_floats:
                for _ in range(n_steps):
                    yield f
                    f = np.nextafter(f, float('-inf'))

        for x in float_generator():
            t = qutypes.TimeType.from_float(x)
            t2x = float(t)
            self.assertEqual(x, t2x)
            self.assertGreater(t2x, np.nextafter(x, float('-inf')))
            self.assertLess(t2x, np.nextafter(x, float('inf')))

