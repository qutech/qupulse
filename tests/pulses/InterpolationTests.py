import unittest
import numpy as np

from pulses.Interpolation import LinearInterpolationStrategy, HoldInterpolationStrategy, JumpInterpolationStrategy
from sqlalchemy.engine.strategies import strategies

class InterpolationTest(unittest.TestCase):

    def test_linear_interpolation(self):
        start = (-1, -1)
        end = (3,3)
        t = np.arange(-1, 4, dtype=float)
        strat = LinearInterpolationStrategy()
        result = strat(start, end, t)
        self.assertTrue(all(t ==  result))
        # TODO: Discussion: May start > end?
        
    def test_hold_interpolation(self):
        start = (-1, -1)
        end = (3,3)
        t = np.linspace(-1,3,100)
        strat = HoldInterpolationStrategy()
        result = strat(start, end, t)
        self.assertTrue(all(result == -1))
        self.assertRaises(ValueError, strat, end, start, t)

    def test_jump_interpolation(self):
        start = (-1, -1)
        end = (3,3)
        t = np.linspace(-1,3,100)
        strat = JumpInterpolationStrategy()
        result = strat(start, end, t)
        self.assertTrue(all(result == 3))
        self.assertRaises(ValueError, strat, end, start, t)
    
    def test_repr_str(self):
        #Test hash
        strategies = {LinearInterpolationStrategy():("linear","<Linear Interpolation>"),
                      HoldInterpolationStrategy():  ("hold",  "<Hold Interpolation>"),
                      JumpInterpolationStrategy():  ("jump",  "<Jump Interpolation>")}
        
        for strategy in strategies:
            repr_ = strategies[strategy][1]
            str_ = strategies[strategy][0]
            self.assertEqual(repr(strategy), repr_)
            self.assertEqual(str(strategy), str_)
        self.assertTrue(LinearInterpolationStrategy()!=HoldInterpolationStrategy())
        self.assertTrue(LinearInterpolationStrategy()!=JumpInterpolationStrategy())
        self.assertTrue(JumpInterpolationStrategy()!=HoldInterpolationStrategy())
        
         
        
        
if __name__ == "__main__":
    unittest.main(verbosity=2)