import unittest
import os
import sys

"""Change the path as we were in the similar path in the src directory"""
srcPath = "src".join(os.path.dirname(os.path.abspath(__file__)).rsplit('tests',1))
sys.path.insert(0,srcPath)

from PulseTemplate import PulseTemplate
from SequencePulseTemplate import SequencePulseTemplate, Mapping, DoubleMappingException

class MappingTest(unittest.TestCase):
    def _dummy(a):
        return a*2
    def test_add_mapping_function(self):
        m = Mapping()
        m.add_mapping_function("a","b",_dummy)
        self.assertRaises(DoubleMappingException,m.add_mapping_function,"c","b",_dummy)
        
if __name__ == "__main__":
    unittest.main(verbosity=2)