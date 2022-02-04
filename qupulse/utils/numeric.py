from typing import Tuple, Type
from numbers import Rational
from math import gcd
from operator import le
from functools import partial

import sympy


def lcm(a: int, b: int):
    """least common multiple"""
    return a * b // gcd(a, b)


def smallest_factor_ge(n: int, min_factor: int, brute_force: int = 5):
    """Find the smallest factor of n that is greater or equal min_factor

    Args:
        n: number to factorize
        min_factor: factor must be larger this
        brute_force: range(min_factor, min(min_factor + brute_force)) is probed by brute force

    Returns:
        Smallest factor of n that is greater or equal min_factor
    """
    assert min_factor <= n

    # this shortcut force shortcut costs 1us max
    for factor in range(min_factor, min(min_factor + brute_force, n)):
        if n % factor == 0:
            return factor
    else:
        return min(filter(partial(le, min_factor),
                          sympy.ntheory.divisors(n, generator=True)))


def _approximate_int(alpha_num: int, d_num: int, den: int) -> Tuple[int, int]:
    """Find the best fraction approximation of alpha_num / den with an error smaller d_num / den. Best means the
    fraction with the smallest denominator.

    Algorithm from https://link.springer.com/content/pdf/10.1007%2F978-3-540-72914-3.pdf

    Args:s
        alpha_num: Numerator of number to approximate. 0 < alpha_num < den
        d_num: Numerator of allowed absolute error.
        den: Denominator of both numbers above.

    Returns:
        (numerator, denominator)
    """
    assert 0 < alpha_num < den

    lower_num = alpha_num - d_num
    upper_num = alpha_num + d_num

    p_a, q_a = 0, 1
    p_b, q_b = 1, 1

    p_full, q_full = p_b, q_b

    to_left = True

    while True:

        # compute the number of steps to the left
        x_num = den * p_b - alpha_num * q_b
        x_den = -den * p_a + alpha_num * q_a
        x = (x_num + x_den - 1) // x_den  # ceiling division

        p_full += x * p_a
        q_full += x * q_a

        p_prev = p_full - p_a
        q_prev = q_full - q_a

        # check whether we have a valid approximation
        if (q_full * lower_num < p_full * den < q_full * upper_num or
                q_prev * lower_num < p_prev * den < q_prev * upper_num):
            bound_num = upper_num if to_left else lower_num

            k_num = den * p_b - bound_num * q_b
            k_den = bound_num * q_a - den * p_a
            k = (k_num // k_den) + 1

            return p_b + k * p_a, q_b + k * q_a

        # update the interval
        p_a = p_prev
        q_a = q_prev

        p_b = p_full
        q_b = q_full

        to_left = not to_left


def approximate_rational(x: Rational, abs_err: Rational, fraction_type: Type[Rational]) -> Rational:
    """Return the fraction with the smallest denominator in (x - abs_err, x + abs_err)"""
    if abs_err <= 0:
        raise ValueError('abs_err must be > 0')

    xp, xq = x.numerator, x.denominator
    if xq == 1:
        return x

    dp, dq = abs_err.numerator, abs_err.denominator

    # separate integer part. alpha_num is guaranteed to be < xq
    n, alpha_num = divmod(xp, xq)

    # find common denominator of alpha_num / xq and dp / dq
    den = lcm(xq, dq)
    alpha_num = alpha_num * den // xq
    d_num = dp * den // dq

    if alpha_num < d_num:
        p, q = 0, 1
    else:
        p, q = _approximate_int(alpha_num, d_num, den)

    return fraction_type(p + n * q, q)


def approximate_double(x: float, abs_err: float, fraction_type: Type[Rational]) -> Rational:
    """Return the fraction with the smallest denominator in (x - abs_err, x + abs_err)."""
    return approximate_rational(fraction_type(x), fraction_type(abs_err), fraction_type=fraction_type)
