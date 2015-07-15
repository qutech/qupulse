import unittest
import os
import sys

srcPath = os.path.dirname(os.path.abspath(__file__)).rsplit('tests',1)[0] + 'src'
sys.path.insert(0,srcPath)

from pulses.PulseTemplate import PulseTemplate
from pulses.SequencePulseTemplate import SequencePulseTemplate, Mapping, DoubleMappingException

class MappingTest(unittest.TestCase):
    def _dummy(a):
        return a*2
    def test_add_mapping_function(self):
        m = Mapping()
        m.add_mapping_function("a","b",self._dummy)
        self.assertRaises(DoubleMappingException,m.add_mapping_function,"c","b",self._dummy)
        
if __name__ == "__main__":
    unittest.main(verbosity=2)