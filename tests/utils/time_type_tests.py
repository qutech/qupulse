import unittest
import fractions
import random

import numpy as np
import sympy
import gmpy2

from qupulse.utils.types import TimeType, time_from_float


def assert_from_fraction_works(test: unittest.TestCase, time_type):
    t = time_type.from_fraction(43, 12)
    test.assertIsInstance(t, time_type)
    test.assertEqual(t, fractions.Fraction(43, 12))


def assert_from_float_exact_works(test: unittest.TestCase, time_type):
    test.assertEqual(time_type.from_float(123 / 931, 0),
                     fractions.Fraction(123 / 931))


def assert_fraction_time_from_float_with_precision_works(test: unittest.TestCase, time_type):
    test.assertEqual(time_type.from_float(1000000 / 1000001, 1e-5),
                     fractions.Fraction(1))
    test.assertEqual(time_type.from_float(2.50000000000008, absolute_error=1e-10),
                     time_type.from_fraction(5, 2))
    test.assertEqual(time_type.from_float(9926.666666667, absolute_error=1e-9),
                     time_type.from_fraction(29780, 3))


def assert_from_float_no_extra_args_works(test: unittest.TestCase, time_type):
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
        test.assertEqual(x, t2x)
        test.assertGreater(t, np.nextafter(x, float('-inf')))
        test.assertLess(t, np.nextafter(x, float('inf')))


def assert_try_from_any_works(test: unittest.TestCase, time_type):
    try_from_any = time_type._try_from_any

    # these duck types are here because isinstance(<gmpy2 obj>, numbers.<NumberType>) is version dependent
    class DuckTypeWrapper:
        def __init__(self, value):
            self.value = value

        def __repr__(self):
            return f'{type(self)}({self.value})'

    class DuckInt(DuckTypeWrapper):
        def __int__(self):
            return int(self.value)

    class DuckFloat(DuckTypeWrapper):
        def __float__(self):
            return float(self.value)

    class DuckIntFloat(DuckFloat):
        def __int__(self):
            return int(self.value)

    class DuckRational:
        def __init__(self, numerator, denominator):
            self.numerator = numerator
            self.denominator = denominator

        def __repr__(self):
            return f'{type(self)}({self.numerator}, {self.denominator})'

    for_array_tests = []

    signed_int_types = [int, sympy.Integer, np.int8, np.int16, np.int32, np.int64, DuckInt, DuckIntFloat, gmpy2.mpz]

    for s_t in signed_int_types:
        for val in (1, 17, -17):
            any_val = s_t(val)
            expected_val = time_type.from_fraction(int(val), 1)
            test.assertEqual(expected_val, try_from_any(any_val))
            for_array_tests.append((expected_val, any_val))

    unsigned_int_types = [np.uint8, np.uint16, np.uint32, np.uint]
    for u_t in unsigned_int_types:
        for val in (1, 17):
            any_val = u_t(val)
            expected_val = time_type.from_fraction(int(val), 1)
            test.assertEqual(expected_val, try_from_any(any_val))
            for_array_tests.append((expected_val, any_val))

    rational_types = [fractions.Fraction, sympy.Rational, time_type.from_fraction, DuckRational]
    if gmpy2:
        rational_types.append(gmpy2.mpq)
    for r_t in rational_types:
        for num, den in ((1, 3), (-3, 8), (17, 5)):
            any_val = r_t(num, den)
            expected_val = time_type.from_fraction(num, den)
            test.assertEqual(expected_val, try_from_any(any_val))
            for_array_tests.append((expected_val, any_val))

    float_types = [float, sympy.Float, DuckFloat, DuckIntFloat]
    if gmpy2:
        float_types.append(gmpy2.mpfr)
    for f_t in float_types:
        for val in (3.4, -3., 1.):
            any_val = f_t(val)
            expected_val = time_type.from_float(val)
            test.assertEqual(expected_val, try_from_any(any_val))
            for_array_tests.append((expected_val, any_val))

    arr = np.array(for_array_tests, dtype='O')
    any_arr = arr[:, 1]
    expected_arr = arr[:, 0]
    np.testing.assert_equal(expected_arr, try_from_any(any_arr))


def assert_comparisons_work(test: unittest.TestCase, time_type):
    tt = time_type.from_float(1.1)

    test.assertLess(tt, 4)
    test.assertLess(tt, 4.)
    test.assertLess(tt, time_type.from_float(4.))
    test.assertLess(tt, float('inf'))

    test.assertLessEqual(tt, 4)
    test.assertLessEqual(tt, 4.)
    test.assertLessEqual(tt, time_type.from_float(4.))
    test.assertLessEqual(tt, float('inf'))

    test.assertGreater(tt, 1)
    test.assertGreater(tt, 1.)
    test.assertGreater(tt, time_type.from_float(1.))
    test.assertGreater(tt, float('-inf'))

    test.assertGreaterEqual(tt, 1)
    test.assertGreaterEqual(tt, 1.)
    test.assertGreaterEqual(tt, time_type.from_float(1.))
    test.assertGreaterEqual(tt, float('-inf'))

    test.assertFalse(tt == float('nan'))
    test.assertFalse(tt <= float('nan'))
    test.assertFalse(tt >= float('nan'))
    test.assertFalse(tt < float('nan'))
    test.assertFalse(tt > float('nan'))


class TestTimeType(unittest.TestCase):
    """Tests the TimeType class. The layout of this test is in this way for historic reasons, i.e. to allow testing
    different internal representations for the time type. Right now only gmpy.mpq is implemented and tested."""

    def test_non_finite_float(self):
        with self.assertRaisesRegex(ValueError, 'Cannot represent'):
            TimeType.from_float(float('inf'))
        with self.assertRaisesRegex(ValueError, 'Cannot represent'):
            TimeType.from_float(float('-inf'))
        with self.assertRaisesRegex(ValueError, 'Cannot represent'):
            TimeType.from_float(float('nan'))

    def test_fraction_time_from_fraction(self):
        assert_from_fraction_works(self, TimeType)

    def test_fraction_time_from_float_exact(self):
        assert_from_float_exact_works(self, TimeType)

    def test_fraction_time_from_float_with_precision(self):
        assert_fraction_time_from_float_with_precision_works(self, TimeType)

    def test_from_float_no_extra_args(self):
        assert_from_float_exact_works(self, TimeType)

    def test_from_float_exceptions(self):
        with self.assertRaisesRegex(ValueError, '> 0'):
            time_from_float(.8, -1)

        with self.assertRaisesRegex(ValueError, '<= 1'):
            time_from_float(.8, 2)

    def test_try_from_any(self):
        assert_try_from_any_works(self, TimeType)

    def test_comparisons_work(self):
        assert_comparisons_work(self, TimeType)


def get_some_floats(seed=42, n=1000):
    rand = random.Random(seed)
    return [rand.random()*100 - 50 for _ in range(n)]


def get_from_float(fs):
    return [time_from_float(f) for f in fs]


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
