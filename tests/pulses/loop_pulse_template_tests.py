import unittest
import os
import sys
from qctoolkit.pulses.loop_pulse_template import LoopPulseTemplate


srcPath = os.path.dirname(os.path.abspath(__file__)).rsplit('tests',1)[0] + 'qctoolkit'
sys.path.insert(0,srcPath)

class GenericLoopPulseTemplateTest(unittest.TestCase):
    pass

if __name__ == "__main__":
    unittest.main(verbosity=2)