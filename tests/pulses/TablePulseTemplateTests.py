import unittest
import os
import sys

srcPath = os.path.dirname(os.path.abspath(__file__)).rsplit('tests',1)[0] + 'src'
sys.path.insert(0,srcPath)

from pulses.TablePulseTemplate import TablePulseTemplate, TableEntry, clean_entries
from pulses.Parameter import ParameterDeclaration, Parameter
from pulses.Interpolation import HoldInterpolationStrategy, LinearInterpolationStrategy, JumpInterpolationStrategy

class TablePulseTemplateTest(unittest.TestCase):

    def test_add_entry_for_interpolation(self) -> None:
        table = TablePulseTemplate()
        strategies = ["linear","hold","jump","hold"]
        for i,strategy in enumerate(strategies):
            table.add_entry(2*(i+1), i+1, strategy)

        self.assertRaises(ValueError, table.add_entry, 1,2, "bar")

    def test_interpolation_strategies(self) -> None:
        table = TablePulseTemplate()
        strategies = ["linear", "hold", "jump"]
        for i,strategy in enumerate(strategies):
            table.add_entry(i, i, strategy)

        manual = [(0,0,LinearInterpolationStrategy()), (1,1,HoldInterpolationStrategy()), (2,2,JumpInterpolationStrategy())]
        self.assertEqual(manual, table.entries)

    def test_string_parameters(self):
        # This code sends the consistency check into infinite recursion
        square = TablePulseTemplate()
        square.add_entry('up', 1)
        square.add_entry('down', 0)

    def test_measurement_windows(self):
        square = TablePulseTemplate(measurement=True)
        square.add_entry(1, 1)
        square.add_entry(3, 0)
        square.add_entry(5, 0)
        windows = square.get_measurement_windows()
        self.assertEqual(windows, [(0,5)])

    def test_measurement_windows_with_parameters(self):
        pulse = TablePulseTemplate(measurement=True)
        pulse.add_entry('length', 0)
        parameters = dict(length=100)
        windows = pulse.get_measurement_windows(parameters)
        self.assertEqual(windows, [(0, 100)])
        
    def test_add_entry_empty_time_is_negative(self) -> None:
        table = TablePulseTemplate()
        self.assertRaises(ValueError, table.add_entry, -2, 0)
        
    def test_add_entry_empty_time_is_0(self) -> None:
        table = TablePulseTemplate()
        table.add_entry(0, 3.1)
        self.assertEqual([(0, 3.1, HoldInterpolationStrategy())], table.entries)
        
    def test_add_entry_empty_time_is_positive(self) -> None:
        table = TablePulseTemplate()
        table.add_entry(2, -254.67)
        self.assertEqual([(0, 0, HoldInterpolationStrategy()), (2, -254.67, HoldInterpolationStrategy())], table.entries)
        
    def test_add_entry_empty_time_is_str(self) -> None:
        table = TablePulseTemplate()
        table.add_entry('t', 0)
        decl = ParameterDeclaration('t', min=0)
        self.assertEqual([(0, 0, HoldInterpolationStrategy()), (decl, 0, HoldInterpolationStrategy())], table.entries)

class CleanEntriesTests(unittest.TestCase):
    def empty_list_test(self):
        self.assertEqual([], clean_entries([]))

    def test_point_removal(self):
        table = TablePulseTemplate()
        table.add_entry(1,5)
        table.add_entry(1.5,5)
        table.add_entry(2,5)
        table.add_entry(3,0)
        clean = clean_entries(table.entries)

        table2 = TablePulseTemplate()
        table2.add_entry(1,5)
        table2.add_entry(2,5)
        table2.add_entry(3,0)

        self.assertEqual(clean, table2.entries)

if __name__ == "__main__":
    unittest.main(verbosity=2)