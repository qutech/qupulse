import unittest
import numpy as np

from pulses.Interpolation import LinearInterpolationStrategy, HoldInterpolationStrategy, JumpInterpolationStrategy

class InterpolationTest(unittest.TestCase):

    def test_linear_interpolation(self):
        start = (-1, -1)
        end = (3,3)
        t = np.arange(-1, 4, dtype=float)
        strat = LinearInterpolationStrategy()
        result = strat(start, end, t)
        self.assertTrue(all(t ==  result))

    def test_hold_interpolation(self):
        start = (-1, -1)
        end = (3,3)
        t = np.linspace(-1,3,100)
        strat = HoldInterpolationStrategy()
        result = strat(start, end, t)
        self.assertTrue(all(result == -1))

    def test_jump_interpolation(self):
        start = (-1, -1)
        end = (3,3)
        t = np.linspace(-1,3,100)
        strat = JumpInterpolationStrategy()
        result = strat(start, end, t)
        self.assertTrue(all(result == 3))
