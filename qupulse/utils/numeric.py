from typing import Tuple, Type, Optional
from numbers import Rational, Real
from math import gcd


def lcm(a: int, b: int):
    """least common multiple"""
    return a * b // gcd(a, b)


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


def are_durations_compatible(first_duration: Real, *other_durations: Real,
                             max_abs_spread=1e-10, max_rel_spread=1e-10) -> Optional[bool]:
    """Durations and maximum allowed spreads must be positive.

    For the durations to be considered compatible, the difference between them must be smaller than at least one of
    the allowed spreads.

    Args:
        first_duration: Singled out duration for performance reasons. Not handled differently by the algorithm.
        *other_durations: Other durations to compare for compatibility
        max_abs_spread: Maximum difference for being considered "compatible", regardless of the magnitude of the input
        max_rel_spread: maximum difference for being considered "compatible", relative to the magnitude of the
          maximum input duration

    Returns:
        True or False if decidable else None
    """
    min_duration = max_duration = first_duration
    for duration in other_durations:
        min_duration = min(min_duration, duration)
        max_duration = max(max_duration, duration)
    assert 0 < max_duration, "At least one duration must be positive"
    # spread = max_duration - min_duration
    # allowed_spread = max(max_rel_spread * max_duration, max_abs_spread)
    are_compatible = max_duration - min_duration < max(max_rel_spread * max_duration, max_abs_spread)
    if are_compatible in (False, True):
        return are_compatible

    # durations are sympy expressions with clear ordering
    elif are_compatible.is_Boolean:
        return bool(are_compatible)

    else:
        # Not decidable
        return None


