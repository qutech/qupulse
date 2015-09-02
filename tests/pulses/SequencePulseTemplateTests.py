import unittest
import os
import sys
import copy

srcPath = os.path.dirname(os.path.abspath(__file__)).rsplit('tests',1)[0] + 'src'
sys.path.insert(0,srcPath)

from pulses.PulseTemplate import PulseTemplate, ParameterNotInPulseTemplateException
from pulses.TablePulseTemplate import TablePulseTemplate, TableEntry
from pulses.SequencePulseTemplate import SequencePulseTemplate, MissingMappingException, UnnecessaryMappingException
from pulses.Interpolation import HoldInterpolationStrategy
from pulses.Parameter import ParameterNotProvidedException

class SequencePulseTemplateTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Setup test data
        self.square = TablePulseTemplate()
        self.square.add_entry('up', 'v', 'hold')
        self.square.add_entry('down', 0, 'hold')
        self.square.add_entry('length', 0)

        self.mapping1 = {}
        self.mapping1['up'] = lambda ps: ps['uptime']
        self.mapping1['down'] = lambda ps: ps['uptime'] + ps['length']
        self.mapping1['v'] = lambda ps: ps['voltage']
        self.mapping1['length'] = lambda ps: ps['pulse-length'] * 0.5

        self.outer_parameters = ['uptime', 'length', 'pulse-length', 'voltage']

        self.parameters = {}
        self.parameters['uptime'] = 5
        self.parameters['length'] = 10
        self.parameters['pulse-length'] = 100
        self.parameters['voltage'] = 10

        self.sequence = SequencePulseTemplate([(self.square, self.mapping1)], self.outer_parameters)

    def test_missing_mapping(self):
        mapping = self.mapping1
        mapping.pop('v')

        subtemplates = [(self.square, mapping)]
        with self.assertRaises(MissingMappingException):
            SequencePulseTemplate(subtemplates, self.outer_parameters)

    def test_unnecessary_parameters(self):
        mapping = self.mapping1
        mapping['unnecessary'] = lambda ps: ps['voltage']

        subtemplates = [(self.square, mapping)]
        with self.assertRaises(UnnecessaryMappingException):
            SequencePulseTemplate(subtemplates, self.outer_parameters)

    def test_simple_instantiating(self):
        subtemplates = [(self.square, self.mapping1)]
        sequence = SequencePulseTemplate(subtemplates, self.outer_parameters)

        entries = sequence.get_entries_instantiated(self.parameters)
        entries_manual = [TableEntry(0,0,HoldInterpolationStrategy()),
                          TableEntry(5,10,HoldInterpolationStrategy()),
                          TableEntry(15,0,HoldInterpolationStrategy()),
                          TableEntry(50,0,HoldInterpolationStrategy())]
        self.assertEqual(entries, entries_manual)

    def test_nested_instantiating(self):
        mapping = {}
        mapping['uptime'] = lambda ps: ps['up']
        mapping['pulse-length'] = lambda ps: 100
        mapping['length'] = lambda ps: 10
        mapping['voltage'] = lambda ps: ps['voltage']

        outer_parameters = ['up', 'voltage']
        subtemplates = [(self.sequence, mapping),
                        (self.sequence, mapping)]
        sequence = SequencePulseTemplate(subtemplates, outer_parameters)
        parameters = dict(up=20, voltage=10)
        entries = sequence.get_entries_instantiated(parameters)
        entries_manual = [TableEntry(0,0,HoldInterpolationStrategy()),
                          TableEntry(20,10,HoldInterpolationStrategy()),
                          TableEntry(30,0,HoldInterpolationStrategy()),
                          TableEntry(50,0,HoldInterpolationStrategy()),
                          TableEntry(70,10,HoldInterpolationStrategy()),
                          TableEntry(80,0,HoldInterpolationStrategy()),
                          TableEntry(100,0,HoldInterpolationStrategy())]
        self.assertEqual(entries, entries_manual)

    def test_instantiating_incomplete_parameters(self):
        parameters = copy.copy(self.parameters)
        parameters.pop('uptime')

        with self.assertRaises(ParameterNotProvidedException):
            self.sequence.get_entries_instantiated(parameters)

    def test_instantiating_unnecessary_parameters(self):
        parameters = copy.copy(self.parameters)
        parameters['extra'] = 300
        with self.assertRaises(ParameterNotInPulseTemplateException):
            self.sequence.get_entries_instantiated(parameters)

if __name__ == "__main__":
    unittest.main(verbosity=2)