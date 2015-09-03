import unittest
import os
import sys

srcPath = os.path.dirname(os.path.abspath(__file__)).rsplit('tests',1)[0] + 'src'
sys.path.insert(0,srcPath)

from pulses.TablePulseTemplate import TablePulseTemplate, TableEntry, clean_entries
from pulses.Parameter import ParameterDeclaration, Parameter
from pulses.Interpolation import HoldInterpolationStrategy, LinearInterpolationStrategy, JumpInterpolationStrategy

class TablePulseTemplateTest(unittest.TestCase):

    def test_add_entry_known_interpolation_strategies(self) -> None:
        table = TablePulseTemplate()
        strategies = ["linear", "hold", "jump"]
        for i,strategy in enumerate(strategies):
            table.add_entry(i, i, strategy)

        manual = [(0,0,LinearInterpolationStrategy()), (1,1,HoldInterpolationStrategy()), (2,2,JumpInterpolationStrategy())]
        self.assertEqual(manual, table.entries)

    def test_add_entry_unknown_interpolation_strategy(self) -> None:
        table = TablePulseTemplate()
        self.assertRaises(ValueError, table.add_entry, 0, 0, 'foo')
        self.assertRaises(ValueError, table.add_entry, 3.2, 0, 'foo')

    def test_add_entry_for_interpolation(self) -> None:
        table = TablePulseTemplate()
        strategies = ["linear","hold","jump","hold"]
        for i,strategy in enumerate(strategies):
            table.add_entry(2*(i+1), i+1, strategy)

        self.assertRaises(ValueError, table.add_entry, 1,2, "bar")

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
        self.assertFalse(table.entries)
        self.assertFalse(table.parameter_declarations)
        self.assertFalse(table.parameter_names)
        
    def test_add_entry_empty_time_is_0(self) -> None:
        table = TablePulseTemplate()
        table.add_entry(0, 3.1)
        self.assertEqual([(0, 3.1, HoldInterpolationStrategy())], table.entries)
        self.assertFalse(table.parameter_names)
        self.assertFalse(table.parameter_declarations)

    def test_add_entry_empty_time_is_0_voltage_is_parameter(self) -> None:
        table = TablePulseTemplate()
        table.add_entry(0, 'foo')
        decl = ParameterDeclaration('foo')
        self.assertEqual([(0, decl, HoldInterpolationStrategy())], table.entries)
        self.assertEqual({'foo'}, table.parameter_names)
        self.assertEqual({decl}, table.parameter_declarations)
        
    def test_add_entry_empty_time_is_positive(self) -> None:
        table = TablePulseTemplate()
        table.add_entry(2, -254.67)
        self.assertEqual([(0, 0, HoldInterpolationStrategy()), (2, -254.67, HoldInterpolationStrategy())], table.entries)
        self.assertFalse(table.parameter_names)
        self.assertFalse(table.parameter_declarations)
        
    def test_add_entry_empty_time_is_str(self) -> None:
        table = TablePulseTemplate()
        table.add_entry('t', 0)
        decl = ParameterDeclaration('t', min=0)
        self.assertEqual([(0, 0, HoldInterpolationStrategy()), (decl, 0, HoldInterpolationStrategy())], table.entries)
        self.assertEqual({'t'}, table.parameter_names)
        self.assertEqual({decl}, table.parameter_declarations)

    def test_add_entry_empty_time_is_declaration(self) -> None:
        table = TablePulseTemplate()
        decl = ParameterDeclaration('foo')
        table.add_entry(decl, 0)
        decl.min_value = 0
        self.assertEqual([(0, 0, HoldInterpolationStrategy()), (decl, 0, HoldInterpolationStrategy())], table.entries)
        self.assertEqual({'foo'}, table.parameter_names)
        self.assertEqual({decl}, table.parameter_declarations)

    def test_add_entry_time_float_after_float(self) -> None:
        table = TablePulseTemplate()
        table.add_entry(1.2, -3.8)
        # expect ValueError if next float is smaller or equal than previous
        self.assertRaises(ValueError, table.add_entry, 0.423, 0)
        self.assertRaises(ValueError, table.add_entry, 1.2, 0)
        # adding a higher value as next entry should work
        table.add_entry(3.7, 1.34875)
        self.assertEqual([(0, 0, HoldInterpolationStrategy()), (1.2, -3.8, HoldInterpolationStrategy()), (3.7, 1.34875, HoldInterpolationStrategy())], table.entries)
        self.assertFalse(table.parameter_names)
        self.assertFalse(table.parameter_declarations)

    def test_add_entry_time_float_after_declaration_no_bound(self) -> None:
        table = TablePulseTemplate()
        table.add_entry('t', 7.1)
        table.add_entry(2.1, 5.5)
        decl = ParameterDeclaration('t', min=0, max=2.1)
        self.assertEqual([(0, 0, HoldInterpolationStrategy()), (decl, 7.1, HoldInterpolationStrategy()), (2.1, 5.5, HoldInterpolationStrategy())], table.entries)
        self.assertEqual({'t'}, table.parameter_names)
        self.assertEqual({decl}, table.parameter_declarations)

    def test_add_entry_time_float_after_declaration_greater_bound(self) -> None:
        table = TablePulseTemplate()
        decl = ParameterDeclaration('t', max=3.4)
        table.add_entry(decl, 7.1)
        decl.min_value = 0
        self.assertRaises(ValueError, table.add_entry, 2.1, 5.5)
        self.assertEqual([(0, 0, HoldInterpolationStrategy()), (decl, 7.1, HoldInterpolationStrategy())], table.entries)
        self.assertEqual({'t'}, table.parameter_names)
        self.assertEqual({decl}, table.parameter_declarations)

    def test_add_entry_time_float_after_declaration_smaller_bound(self) -> None:
        table = TablePulseTemplate()
        decl = ParameterDeclaration('t', min=1.0, max=1.3)
        table.add_entry(decl, 7.1)
        table.add_entry(2.1, 5.5)
        self.assertEqual([(0, 0, HoldInterpolationStrategy()), (decl, 7.1, HoldInterpolationStrategy()), (2.1, 5.5, HoldInterpolationStrategy())], table.entries)
        self.assertEqual({'t'}, table.parameter_names)
        self.assertEqual({decl}, table.parameter_declarations)

    def test_add_entry_time_parameter_name_in_use_as_voltage(self) -> None:
        table = TablePulseTemplate()
        table.add_entry(0, 'foo')
        foo_decl = ParameterDeclaration('foo')
        self.assertEqual({'foo'}, table.parameter_names)
        self.assertEqual({foo_decl}, table.parameter_declarations)
        self.assertRaises(ValueError, table.add_entry, 'foo', 4.3)
        self.assertEqual({'foo'}, table.parameter_names)
        self.assertEqual({foo_decl}, table.parameter_declarations)
        self.assertEqual([(0, foo_decl, HoldInterpolationStrategy())], table.entries)

    def test_add_entry_time_parmeter_name_in_use_as_time(self) -> None:
        table = TablePulseTemplate()
        table.add_entry('foo', 'bar')
        foo_decl = ParameterDeclaration('foo', min=0)
        bar_decl = ParameterDeclaration('bar')
        self.assertRaises(ValueError, table.add_entry, ParameterDeclaration('foo'), 3.4)
        self.assertEqual([(0, 0, HoldInterpolationStrategy()), (foo_decl, bar_decl, HoldInterpolationStrategy())], table.entries)
        self.assertEqual({'foo', 'bar'}, table.parameter_names)
        self.assertEqual({foo_decl, bar_decl}, table.parameter_declarations)

    def test_add_entry_time_declaration_invalid_bounds(self) -> None:
        table = TablePulseTemplate()
        bar_decl = ParameterDeclaration('bar')
        foo_decl = ParameterDeclaration('foo')
        foo_decl.min_value = bar_decl
        self.assertRaises(ValueError, table.add_entry, foo_decl, 23857.23)
        self.assertRaises(ValueError, table.add_entry, bar_decl, -4967.1)
        self.assertFalse(table.entries)
        self.assertFalse(table.parameter_names)
        self.assertFalse(table.parameter_declarations)

    def test_add_entry_time_declaration_no_bounds_after_float(self) -> None:
        table = TablePulseTemplate()
        table.add_entry(3.2, 92.1)
        table.add_entry('t', 1.2)
        decl = ParameterDeclaration('t', min=3.2)
        self.assertEqual([(0, 0, HoldInterpolationStrategy()), (3.2, 92.1, HoldInterpolationStrategy()), (decl, 1.2, HoldInterpolationStrategy())], table.entries)
        self.assertEqual({'t'}, table.parameter_names)
        self.assertEqual({decl}, table.parameter_declarations)

    def test_add_entry_time_declaration_higher_min_after_float(self) -> None:
        table = TablePulseTemplate()
        table.add_entry(3.2, 92.1)
        decl = ParameterDeclaration('t', min=4.5)
        table.add_entry(decl, 1.2)
        self.assertEqual([(0, 0, HoldInterpolationStrategy()), (3.2, 92.1, HoldInterpolationStrategy()), (decl, 1.2, HoldInterpolationStrategy())], table.entries)
        self.assertEqual({'t'}, table.parameter_names)
        self.assertEqual({decl}, table.parameter_declarations)

    def test_add_entry_time_declaration_lower_min_after_float(self) -> None:
        table = TablePulseTemplate()
        table.add_entry(3.2, 92.1)
        decl = ParameterDeclaration('t', min=0.1)
        self.assertRaises(ValueError, table.add_entry, decl, 1.2)
        self.assertEqual([(0, 0, HoldInterpolationStrategy()), (3.2, 92.1, HoldInterpolationStrategy())], table.entries)
        self.assertFalse(table.parameter_names)
        self.assertFalse(table.parameter_declarations)

    def test_add_entry_time_declaration_after_declaration_no_upper_bound(self) -> None:
        table = TablePulseTemplate()
        table.add_entry('bar', 72.14)
        table.add_entry('foo', 0)
        bar_decl = ParameterDeclaration('bar', min=0)
        foo_decl = ParameterDeclaration('foo')
        foo_decl.min_value = bar_decl
        self.assertEqual([(0, 0, HoldInterpolationStrategy()), (bar_decl, 72.14, HoldInterpolationStrategy()), (foo_decl, 0, HoldInterpolationStrategy())], table.entries)
        self.assertEqual({'bar', 'foo'}, table.parameter_names)
        self.assertEqual({bar_decl, foo_decl}, table.parameter_declarations)

    def test_add_entry_time_declaration_after_declaration_upper_bound(self) -> None:
        table = TablePulseTemplate()
        bar_decl = ParameterDeclaration('bar', min=1, max=2)
        foo_decl = ParameterDeclaration('foo')
        table.add_entry(bar_decl, -3)
        table.add_entry(foo_decl, 0.1)
        foo_decl.min_value = bar_decl
        self.assertEqual([(0, 0, HoldInterpolationStrategy()), (bar_decl, -3, HoldInterpolationStrategy()), (foo_decl, 0.1, HoldInterpolationStrategy())], table.entries)
        self.assertEqual({'foo', 'bar'}, table.parameter_names)
        self.assertEqual({foo_decl, bar_decl}, table.parameter_declarations)

    def test_add_entry_time_declaration_lower_bound_after_declaration_upper_bound(self) -> None:
        table = TablePulseTemplate()
        bar_decl = ParameterDeclaration('bar', min=1, max=2)
        foo_decl = ParameterDeclaration('foo', min=1)
        table.add_entry(bar_decl, -3)
        table.add_entry(foo_decl, 0.1)
        self.assertEqual([(0, 0, HoldInterpolationStrategy()), (bar_decl, -3, HoldInterpolationStrategy()), (foo_decl, 0.1, HoldInterpolationStrategy())], table.entries)
        self.assertEqual({'foo', 'bar'}, table.parameter_names)
        self.assertEqual({foo_decl, bar_decl}, table.parameter_declarations)

    def test_add_entry_time_declaration_lower_bound_after_declaration_no_upper_bound(self) -> None:
        table = TablePulseTemplate()
        bar_decl = ParameterDeclaration('bar', min=1)
        foo_decl = ParameterDeclaration('foo', min=1)
        table.add_entry(bar_decl, -3)
        table.add_entry(foo_decl, 0.1)
        self.assertEqual([(0, 0, HoldInterpolationStrategy()), (bar_decl, -3, HoldInterpolationStrategy()), (foo_decl, 0.1, HoldInterpolationStrategy())], table.entries)
        self.assertEqual({'foo', 'bar'}, table.parameter_names)
        self.assertEqual({foo_decl, bar_decl}, table.parameter_declarations)

    def test_add_entry_time_declaration_lower_bound_too_small_after_declaration_no_upper_bound(self) -> None:
        table = TablePulseTemplate()
        bar_decl = ParameterDeclaration('bar', min=1)
        foo_decl = ParameterDeclaration('foo', min=0)
        table.add_entry(bar_decl, -3)
        self.assertRaises(ValueError, table.add_entry, foo_decl, 0.1)
        self.assertEqual([(0, 0, HoldInterpolationStrategy()), (bar_decl, -3, HoldInterpolationStrategy())], table.entries)
        self.assertEqual({'bar'}, table.parameter_names)
        self.assertEqual({bar_decl}, table.parameter_declarations)

    def test_add_entry_time_declaration_no_lower_bound_upper_bound_too_small_after_declaration(self) -> None:
        table = TablePulseTemplate()
        bar_decl = ParameterDeclaration('bar', min=1, max=2)
        foo_decl = ParameterDeclaration('foo', max=1)
        table.add_entry(bar_decl, -3)
        table.add_entry(foo_decl, 0.1)
        foo_decl.min_value = bar_decl
        self.assertEqual([(0, 0, HoldInterpolationStrategy()), (bar_decl, -3, HoldInterpolationStrategy()), (foo_decl, 0.1, HoldInterpolationStrategy())], table.entries)
        self.assertEqual({'bar', 'foo'}, table.parameter_names)
        self.assertEqual({bar_decl, foo_decl}, table.parameter_declarations)

    def test_add_entry_time_declaration_lower_bound_upper_bound_too_small_after_declaration(self) -> None:
        table = TablePulseTemplate()
        bar_decl = ParameterDeclaration('bar', min=1, max=2)
        foo_decl = ParameterDeclaration('foo', min=1, max=1.5)
        table.add_entry(bar_decl, -3)
        self.assertRaises(ValueError, table.add_entry, foo_decl, 0.1)
        self.assertEqual([(0, 0, HoldInterpolationStrategy()), (bar_decl, -3, HoldInterpolationStrategy())], table.entries)
        self.assertEqual({'bar'}, table.parameter_names)
        self.assertEqual({bar_decl}, table.parameter_declarations)

    def test_is_interruptable(self) -> None:
        self.assertFalse(TablePulseTemplate().is_interruptable)



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