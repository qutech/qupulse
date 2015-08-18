import unittest
import os
import sys

srcPath = os.path.dirname(os.path.abspath(__file__)).rsplit('tests',1)[0] + 'src'
sys.path.insert(0,srcPath)

from pulses.TablePulseTemplate import TablePulseTemplate
from pulses.Parameter import ParameterDeclaration, Parameter

class TablePulseTemplateTest(unittest.TestCase):

    def test_add_entry(self) -> None:
        table = TablePulseTemplate()
        
        self.assertRaises(ValueError, table.add_entry, -2, 0)
        table.add_entry(2, 2.1)
        self.assertEqual([(0, 0), (2, 2.1)], table.entries)
        self.assertRaises(ValueError, table.add_entry, 1.3, 763)
        table.add_entry('foo', -2)
        self.assertEqual([(0, 0), (2, 2.1), (ParameterDeclaration('foo'), -2)], table.entries)

if __name__ == "__main__":
    unittest.main(verbosity=2)