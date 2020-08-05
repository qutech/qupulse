import sys
import unittest
import builtins
import contextlib
import importlib
import fractions
import random
import math
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
    """The fallback test is here for convenience while developing. The fallback is also tested by the CI explicitly"""

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

    def test_non_finite_float(self):
        with self.assertRaisesRegex(ValueError, 'Cannot represent'):
            qutypes.TimeType.from_float(float('inf'))
        with self.assertRaisesRegex(ValueError, 'Cannot represent'):
            qutypes.TimeType.from_float(float('-inf'))
        with self.assertRaisesRegex(ValueError, 'Cannot represent'):
            qutypes.TimeType.from_float(float('nan'))

    def assert_self_init_works(self, time_type):
        t = time_type.from_fraction(1, 3)
        self.assertIs(t._value, time_type(t)._value)

    def test_self_init(self):
        self.assert_self_init_works(self.fallback_qutypes.TimeType)
        self.assert_self_init_works(qutypes.TimeType)

    def test_fraction_fallback(self):
        self.assertIs(fractions.Fraction, self.fallback_qutypes.TimeType._InternalType)

    def assert_from_fraction_works(self, time_type):
        t = time_type.from_fraction(43, 12)
        self.assertIsInstance(t, time_type)
        self.assertEqual(t, fractions.Fraction(43, 12))

    def test_fraction_time_from_fraction(self):
        self.assert_from_fraction_works(qutypes.TimeType)

    @unittest.skipIf(gmpy2 is None, "fallback already tested")
    def test_fraction_time_from_fraction_fallback(self):
        self.assert_from_fraction_works(self.fallback_qutypes.TimeType)

    def assert_from_float_exact_works(self, time_type):
        self.assertEqual(time_type.from_float(123 / 931, 0),
                         fractions.Fraction(123 / 931))

    def test_fraction_time_from_float_exact(self):
        self.assert_from_float_exact_works(qutypes.TimeType)

    @unittest.skipIf(gmpy2 is None, "fallback already tested")
    def test_fraction_time_from_float_exact_fallback(self):
        self.assert_from_float_exact_works(self.fallback_qutypes.TimeType)

    def assert_fraction_time_from_float_with_precision_works(self, time_type):
        self.assertEqual(time_type.from_float(1000000 / 1000001, 1e-5),
                         fractions.Fraction(1))
        self.assertEqual(time_type.from_float(2.50000000000008, absolute_error=1e-10),
                         time_type.from_fraction(5, 2))
        self.assertEqual(time_type.from_float(9926.666666667, absolute_error=1e-9),
                         time_type.from_fraction(29780, 3))

    def test_fraction_time_from_float_with_precision(self):
        self.assert_fraction_time_from_float_with_precision_works(qutypes.TimeType)

    @unittest.skipIf(gmpy2 is None, "fallback already tested")
    def test_fraction_time_from_float_with_precision_fallback(self):
        self.assert_fraction_time_from_float_with_precision_works(self.fallback_qutypes.TimeType)

    def assert_from_float_no_extra_args_works(self, time_type):
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
            t = time_type.from_float(x)
            t2x = float(t)
            self.assertEqual(x, t2x)
            self.assertGreater(t, np.nextafter(x, float('-inf')))
            self.assertLess(t, np.nextafter(x, float('inf')))

    def test_from_float_no_extra_args(self):
        self.assert_from_float_exact_works(qutypes.TimeType)

    @unittest.skipIf(gmpy2 is None, "fallback already tested")
    def test_from_float_no_extra_args_fallback(self):
        self.assert_from_float_exact_works(self.fallback_qutypes.TimeType)

    def test_from_float_exceptions(self):
        with self.assertRaisesRegex(ValueError, '> 0'):
            qutypes.time_from_float(.8, -1)

        with self.assertRaisesRegex(ValueError, '<= 1'):
            qutypes.time_from_float(.8, 2)

    def assert_comparisons_work(self, time_type):
        tt = time_type.from_float(1.1)

        self.assertLess(tt, 4)
        self.assertLess(tt, 4.)
        self.assertLess(tt, time_type.from_float(4.))
        self.assertLess(tt, float('inf'))

        self.assertLessEqual(tt, 4)
        self.assertLessEqual(tt, 4.)
        self.assertLessEqual(tt, time_type.from_float(4.))
        self.assertLessEqual(tt, float('inf'))

        self.assertGreater(tt, 1)
        self.assertGreater(tt, 1.)
        self.assertGreater(tt, time_type.from_float(1.))
        self.assertGreater(tt, float('-inf'))

        self.assertGreaterEqual(tt, 1)
        self.assertGreaterEqual(tt, 1.)
        self.assertGreaterEqual(tt, time_type.from_float(1.))
        self.assertGreaterEqual(tt, float('-inf'))

        self.assertFalse(tt == float('nan'))
        self.assertFalse(tt <= float('nan'))
        self.assertFalse(tt >= float('nan'))
        self.assertFalse(tt < float('nan'))
        self.assertFalse(tt > float('nan'))

    def test_comparisons_work(self):
        self.assert_comparisons_work(qutypes.TimeType)

    @unittest.skipIf(gmpy2 is None, "fallback already tested")
    def test_comparisons_work_fallback(self):
        self.assert_comparisons_work(self.fallback_qutypes.TimeType)

    def assert_simple_arithmetic_work(self, time_type):
        t1 = time_type.from_fraction(19, 3)
        as_frac = fractions.Fraction(19, 3)

        self.assertEqual(math.ceil(as_frac), math.ceil(t1))
        self.assertEqual(math.floor(as_frac), math.floor(t1))
        self.assertEqual(math.trunc(as_frac), math.trunc(t1))
        self.assertEqual(3 % as_frac, 3 % t1)
        self.assertEqual(-as_frac, -t1)
        self.assertIs(t1, +t1)
        self.assertEqual(as_frac**3, t1**3)
        self.assertEqual(3 ** as_frac, 3 ** t1)
        self.assertEqual(3 / as_frac, 3 / t1)
        self.assertEqual(as_frac // 3, t1 // 3)
        self.assertEqual(3 // as_frac, 3 // t1)

    def test_simple_arithmetic(self):
        self.assert_simple_arithmetic_work(qutypes.TimeType)
        self.assert_simple_arithmetic_work(self.fallback_qutypes.TimeType)

    def test_time_from_fraction(self):
        self.assertEqual(qutypes.time_from_fraction(1, 3), qutypes.TimeType.from_fraction(1, 3))


def get_some_floats(seed=42, n=1000):
    rand = random.Random(seed)
    return [rand.random()*100 - 50 for _ in range(n)]


def get_from_float(fs):
    return [qutypes.time_from_float(f) for f in fs]


def do_additions(xs, ys):
    for x, y in zip(xs, ys):
        _ = x + y


def do_multiplications(xs, ys):
    for x, y in zip(xs, ys):
        _ = x * y


def test_time_type_from_float_performance(benchmark):
    benchmark(get_from_float, get_some_floats())


def test_time_type_addition_performance(benchmark):
    values = get_from_float(get_some_floats())
    benchmark(do_additions, values, values)


def test_time_type_addition_with_float_performance(benchmark):

    benchmark(do_additions,
              get_from_float(get_some_floats(seed=42)),
              get_some_floats(seed=43))


def test_time_type_mul_performance(benchmark):
    values = get_from_float(get_some_floats(seed=42))
    benchmark(do_multiplications, values, values)
