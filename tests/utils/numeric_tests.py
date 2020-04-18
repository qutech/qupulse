import unittest
from typing import Callable, List, Iterator, Tuple
import random
from fractions import Fraction
from collections import deque
from itertools import islice

from qupulse.utils.numeric import approximate_rational, approximate_double


def stern_brocot_sequence() -> Iterator[int]:
    sb = deque([1, 1])
    while True:
        sb += [sb[0] + sb[1], sb[1]]
        yield sb.popleft()


def stern_brocot_tree(depth: int) -> List[Fraction]:
    """see wikipedia article"""
    fractions = [Fraction(0), Fraction(1)]

    seq = stern_brocot_sequence()
    next(seq)

    for n in range(depth):
        for _ in range(n + 1):
            p = next(seq)
            q = next(seq)
            fractions.append(Fraction(p, q))

    return sorted(fractions)


def window(iterable, n):
    assert n > 0
    it = iter(iterable)
    state = deque(islice(it, 0, n - 1), maxlen=n)
    for new_element in it:
        state.append(new_element)
        yield tuple(state)
        state.popleft()


def uniform_without_bounds(rng, a, b):
    result = a
    while not a < result < b:
        result = rng.uniform(a, b)
    return result


def generate_test_pairs(depth: int, seed) -> Iterator[Tuple[Tuple[Fraction, Fraction], Fraction]]:
    rng = random.Random(seed)
    tree = stern_brocot_tree(depth)
    extended_tree = [float('-inf')] + tree + [float('inf')]

    # values map to themselves
    for a, b, c in window(tree, 3):
        err = min(b - a, c - b)
        yield (b, err), b

    for prev, a, b, upcom in zip(extended_tree, extended_tree[1:], extended_tree[2:], extended_tree[3:]):
        mid = (a + b) / 2

        low = Fraction(uniform_without_bounds(rng, a, mid))
        err = min(mid - a, low - prev)
        yield (low, err), a

        high = Fraction(uniform_without_bounds(rng, mid, b))
        err = min(b - mid, upcom - high)
        yield (high, err), b


class ApproximationTests(unittest.TestCase):
    def test_approximate_rational(self):
        """Use Stern-Brocot tree and rng to generate test cases where we know the result"""
        depth = 70  # equivalent to 7457 test cases
        test_pairs = list(generate_test_pairs(depth, seed=42))

        for offset in (-2, -1, 0, 1, 2):
            for (x, abs_err), result in test_pairs:
                expected = result + offset
                result = approximate_rational(x + offset, abs_err, Fraction)
                self.assertEqual(expected, result)

    def test_approximate_double(self):
        test_values = [
            ((.1, .05), Fraction(1, 7)),
            ((.12, .005), Fraction(2, 17)),
            ((.15, .005), Fraction(2, 13)),
            # .111_111_12, 0.000_000_005
            ((.11111112, 0.000000005), Fraction(888890, 8000009)),
            ((.111125, 0.0000005), Fraction(859, 7730)),
            ((2.50000000008, .1), Fraction(5, 2))
        ]

        for (x, err), expected in test_values:
            result = approximate_double(x, err, Fraction)
            self.assertEqual(expected, result, msg='{x} ± {err} results in {result} '
                                                   'which is not the expected {expected}'.format(x=x, err=err,
                                                                                                 result=result,
                                                                                                 expected=expected))
