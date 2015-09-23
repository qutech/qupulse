import unittest
import os
import sys

import numpy as np

srcPath = os.path.dirname(os.path.abspath(__file__)).rsplit('tests',1)[0] + 'src'
sys.path.insert(0,srcPath)

from tests.pulses.SequencingDummies import DummySequencer, DummyInstructionBlock, DummySequencingHardware
from tests.pulses.SerializationDummies import DummySerializer

from pulses.Instructions import EXECInstruction
from pulses.TablePulseTemplate import TablePulseTemplate, clean_entries, ParameterValueIllegalException, TableWaveformData
from pulses.Parameter import ParameterDeclaration, Parameter, ParameterNotProvidedException
from pulses.Interpolation import HoldInterpolationStrategy, LinearInterpolationStrategy, JumpInterpolationStrategy
from pulses.Serializer import Serializer


class DummyParameter(Parameter):

    def __init__(self, value: float = 0, requires_stop: bool = False) -> None:
        super().__init__()
        self.__value = value
        self.__requires_stop = requires_stop

    def get_value(self) -> float:
        return self.__value

    @property
    def requires_stop(self) -> bool:
        return self.__requires_stop

    def get_serialization_data(self, serializer: Serializer) -> None:
            raise NotImplemented()

    @staticmethod
    def deserialize(serializer: Serializer) -> 'DummyParameter':
        raise NotImplemented()


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

    def test_measurement_windows(self) -> None:
        pulse = TablePulseTemplate(measurement=True)
        pulse.add_entry(1, 1)
        pulse.add_entry(3, 0)
        pulse.add_entry(5, 0)
        windows = pulse.get_measurement_windows()
        self.assertEqual([(0,5)], windows)

    def test_no_measurement_windows(self) -> None:
        pulse = TablePulseTemplate(measurement=False)
        pulse.add_entry(1, 1)
        pulse.add_entry(3, 0)
        pulse.add_entry(5, 0)
        windows = pulse.get_measurement_windows()
        self.assertEqual([], windows)

    def test_measurement_windows_with_parameters(self) -> None:
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
        self.maxDiff = None
        bar_decl = ParameterDeclaration('bar', min=1)
        foo_decl = ParameterDeclaration('foo', min=1)
        table.add_entry(bar_decl, -3)
        table.add_entry(foo_decl, 0.1)
        bar_decl.max_value = foo_decl
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
        self.assertRaises(ValueError, table.add_entry, foo_decl, 0.1)
        self.assertEqual([(0, 0, HoldInterpolationStrategy()), (bar_decl, -3, HoldInterpolationStrategy())], table.entries)
        self.assertEqual({'bar'}, table.parameter_names)
        self.assertEqual({bar_decl}, table.parameter_declarations)

    def test_add_entry_time_declaration_lower_bound_upper_bound_too_small_after_declaration(self) -> None:
        table = TablePulseTemplate()
        bar_decl = ParameterDeclaration('bar', min=1, max=2)
        foo_decl = ParameterDeclaration('foo', min=1, max=1.5)
        table.add_entry(bar_decl, -3)
        self.assertRaises(ValueError, table.add_entry, foo_decl, 0.1)
        self.assertEqual([(0, 0, HoldInterpolationStrategy()), (bar_decl, -3, HoldInterpolationStrategy())], table.entries)
        self.assertEqual({'bar'}, table.parameter_names)
        self.assertEqual({bar_decl}, table.parameter_declarations)

    def test_add_entry_voltage_declaration_reuse(self) -> None:
        table = TablePulseTemplate()
        foo_decl = ParameterDeclaration('foo', min=0, max=3.3)
        bar_decl = ParameterDeclaration('bar', min=-3.3, max=1.15)
        table.add_entry(0, foo_decl)
        table.add_entry(1.51, bar_decl)
        table.add_entry(3, 'foo')
        table.add_entry('t', foo_decl)
        t_decl = ParameterDeclaration('t', min=3)
        self.assertEqual([(0, foo_decl, HoldInterpolationStrategy()), (1.51, bar_decl, HoldInterpolationStrategy()),
                          (3, foo_decl, HoldInterpolationStrategy()), (t_decl, foo_decl, HoldInterpolationStrategy())], table.entries)
        self.assertEqual({'foo', 'bar', 't'}, table.parameter_names)
        self.assertEqual({foo_decl, bar_decl, t_decl}, table.parameter_declarations)

    def test_add_entry_voltage_declaration_in_use_as_time(self) -> None:
        table = TablePulseTemplate()
        foo_decl = ParameterDeclaration('foo', min=0, max=2)
        table.add_entry(foo_decl, 0)
        self.assertRaises(ValueError, table.add_entry, 4, foo_decl)
        self.assertEqual([(0, 0, HoldInterpolationStrategy()), (foo_decl, 0, HoldInterpolationStrategy())], table.entries)
        self.assertEqual({'foo'}, table.parameter_names)
        self.assertEqual({foo_decl}, table.parameter_declarations)

    def test_add_entry_time_and_voltage_same_declaration(self) -> None:
        table = TablePulseTemplate()
        self.assertRaises(ValueError, table.add_entry, 'foo', 'foo')
        self.assertFalse(table.entries)
        self.assertFalse(table.parameter_names)
        self.assertFalse(table.parameter_declarations)

    def test_is_interruptable(self) -> None:
        self.assertFalse(TablePulseTemplate().is_interruptable)

    def test_get_entries_instantiated_one_entry_float_float(self) -> None:
        table = TablePulseTemplate()
        table.add_entry(0, 2)
        instantiated_entries = table.get_entries_instantiated({})
        self.assertEqual([(0, 2, HoldInterpolationStrategy())], instantiated_entries)

    def test_get_entries_instantiated_one_entry_float_declaration(self) -> None:
        table = TablePulseTemplate()
        table.add_entry(0, 'foo')
        instantiated_entries = table.get_entries_instantiated({'foo': 2})
        self.assertEqual([(0, 2, HoldInterpolationStrategy())], instantiated_entries)

    def test_get_entries_instantiated_two_entries_float_float_declaration_float(self) -> None:
        table = TablePulseTemplate()
        table.add_entry('foo', -3.1415)
        instantiated_entries = table.get_entries_instantiated({'foo': 2})
        self.assertEqual([(0, 0, HoldInterpolationStrategy()), (2, -3.1415, HoldInterpolationStrategy())], instantiated_entries)

    def test_get_entries_instantiated_two_entries_float_declaraton_declaration_declaration(self) -> None:
        table = TablePulseTemplate()
        table.add_entry(0, 'v1')
        table.add_entry('t', 'v2')
        instantiated_entries = table.get_entries_instantiated({'v1': -5, 'v2': 5, 't': 3})
        self.assertEqual([(0, -5, HoldInterpolationStrategy()), (3, 5, HoldInterpolationStrategy())], instantiated_entries)

    def test_get_entries_instantiated_two_entries_invalid_parameters(self) -> None:
        table = TablePulseTemplate()
        table.add_entry(0, 'v1')
        t_decl = ParameterDeclaration('t', min=1, max=2)
        v2_decl = ParameterDeclaration('v2', min=10, max=30)
        table.add_entry(t_decl, v2_decl)
        self.assertRaises(ParameterValueIllegalException, table.get_entries_instantiated, {'v1': -5, 't': 0, 'v2': 20})
        self.assertRaises(ParameterValueIllegalException, table.get_entries_instantiated, {'v1': -5, 't': 1, 'v2': -20})

    def test_get_entries_instantiated_two_entries_parameter_missing(self) -> None:
        table = TablePulseTemplate()
        t_decl = ParameterDeclaration('t', min=1, max=2)
        v2_decl = ParameterDeclaration('v2', min=10, max=30)
        table.add_entry(t_decl, v2_decl)
        self.assertRaises(ParameterNotProvidedException, table.get_entries_instantiated, {})

    def test_get_entries_instantiated_linked_time_declarations(self) -> None:
        table = TablePulseTemplate()
        foo_decl = ParameterDeclaration('foo', min=1)
        bar_decl = ParameterDeclaration('bar')
        table.add_entry(foo_decl, 'v', 'linear')
        table.add_entry(bar_decl, 0, 'jump')
        instantiated_entries = table.get_entries_instantiated({'v': 2.3, 'foo': 1, 'bar': 4})
        self.assertEqual([(0, 0, HoldInterpolationStrategy()), (1, 2.3, LinearInterpolationStrategy()), (4, 0, JumpInterpolationStrategy())], instantiated_entries)
        self.assertRaises(Exception, table.get_entries_instantiated, {'v': 2.3, 'foo': 1, 'bar': 1})
        self.assertRaises(Exception, table.get_entries_instantiated, {'v': 2.3, 'foo': 1, 'bar': 0.2})

    def test_get_entries_instantiated_unlinked_time_declarations(self) -> None:
        table = TablePulseTemplate()
        foo_decl = ParameterDeclaration('foo', min=1, max=2)
        bar_decl = ParameterDeclaration('bar', min=1.5, max=4)
        table.add_entry(foo_decl, 'v', 'linear')
        table.add_entry(bar_decl, 0, 'jump')
        instantiated_entries = table.get_entries_instantiated({'v': 2.3, 'foo': 1, 'bar': 4})
        self.assertEqual([(0, 0, HoldInterpolationStrategy()), (1, 2.3, LinearInterpolationStrategy()), (4, 0, JumpInterpolationStrategy())], instantiated_entries)
        self.assertRaises(Exception, table.get_entries_instantiated, {'v': 2.3, 'foo': 2, 'bar': 1.5})

    def test_from_array(self) -> None:
        times = np.array([0, 1, 3])
        voltages = np.array([5, 0, 5])
        pulse = TablePulseTemplate.from_array(times, voltages)
        entries = []
        for (time, voltage) in zip(times, voltages):
            entries.append((time, voltage, HoldInterpolationStrategy()))
        self.assertEqual(entries, pulse.entries)


class TablePulseTemplateSerializationTests(unittest.TestCase):

    def setUp(self) -> None:
        self.serializer = DummySerializer(lambda x: dict(name=x.name), lambda x: x.name, lambda x: x['name'])
        self.template = TablePulseTemplate(measurement=True, identifier='foo')
        self.expected_data = dict(type=self.serializer.get_type_identifier(self.template))

    def test_get_serialization_data(self) -> None:
        self.template.add_entry('foo', 2)
        self.template.add_entry('hugo', 'ilse', interpolation='linear')

        self.expected_data['is_measurement_pulse'] = True
        self.expected_data['time_parameter_declarations'] = [dict(name='foo'), dict(name='hugo')]
        self.expected_data['voltage_parameter_declarations'] = [dict(name='ilse')]
        self.expected_data['entries'] = [(0, 0, 'hold'), ('foo', 2, 'hold'), ('hugo', 'ilse', 'linear')]
        self.assertEqual(self.expected_data, self.template.get_serialization_data(self.serializer))

    def test_deserialize(self) -> None:
        data = dict(is_measurement_pulse=True,
                    time_parameter_declarations=[dict(name='hugo'), dict(name='foo')],
                    voltage_parameter_declarations=[dict(name='ilse')],
                    entries=[(0, 0, 'hold'), ('foo', 2, 'hold'), ('hugo', 'ilse', 'linear')],
                    identifier='foo')

        # prepare dependencies for deserialization
        self.serializer.subelements['foo'] = ParameterDeclaration('foo')
        self.serializer.subelements['hugo'] = ParameterDeclaration('hugo')
        self.serializer.subelements['ilse'] = ParameterDeclaration('ilse')

        # deserialize
        template = TablePulseTemplate.deserialize(self.serializer, **data)

        # prepare expected parameter declarations
        self.serializer.subelements['foo'].min_value = 0
        self.serializer.subelements['foo'].max_value = self.serializer.subelements['hugo']
        all_declarations = set(self.serializer.subelements.values())

        # prepare expected entries
        entries = [(0, 0, HoldInterpolationStrategy()),
                   (self.serializer.subelements['foo'], 2, HoldInterpolationStrategy()),
                   (self.serializer.subelements['hugo'], self.serializer.subelements['ilse'], LinearInterpolationStrategy())]

        # compare!
        self.assertEqual(all_declarations, template.parameter_declarations)
        self.assertEqual({'foo', 'hugo', 'ilse'}, template.parameter_names)
        self.assertEqual(entries, template.entries)
        self.assertEqual('foo', template.identifier)


class TablePulseTemplateSequencingTests(unittest.TestCase):

    def test_build_sequence(self) -> None:
        table = TablePulseTemplate()
        foo_decl = ParameterDeclaration('foo', min=1)
        bar_decl = ParameterDeclaration('bar')
        table.add_entry(foo_decl, 'v', 'linear')
        table.add_entry(bar_decl, 0, 'jump')
        parameters = {'v': 2.3, 'foo': 1, 'bar': 4}
        instantiated_entries = tuple(table.get_entries_instantiated(parameters))
        sequencer = DummySequencer(DummySequencingHardware())
        instruction_block = DummyInstructionBlock()
        table.build_sequence(sequencer, parameters, instruction_block)
        waveform = TableWaveformData(instantiated_entries)
        self.assertEqual([waveform], sequencer.hardware.waveforms)
        self.assertEqual(1, len(instruction_block.instructions))
        instruction = instruction_block.instructions[0]
        self.assertIsInstance(instruction, EXECInstruction)
        self.assertEqual(waveform, instruction.waveform)

    def test_build_sequence_empty(self) -> None:
        table = TablePulseTemplate()
        hardware = DummySequencingHardware()
        sequencer = DummySequencer(hardware)
        instruction_block = DummyInstructionBlock()
        table.build_sequence(sequencer, {}, instruction_block)
        self.assertFalse(sequencer.hardware.waveforms)
        self.assertFalse(instruction_block.instructions)

    def test_requires_stop(self) -> None:
        table = TablePulseTemplate()
        foo_decl = ParameterDeclaration('foo', min=1)
        bar_decl = ParameterDeclaration('bar')
        table.add_entry(foo_decl, 'v', 'linear')
        table.add_entry(bar_decl, 0, 'jump')
        test_sets = [(False, {'foo': DummyParameter(0, False), 'bar': DummyParameter(0, False), 'v': DummyParameter(0, False)}),
                     (True, {'foo': DummyParameter(0, True), 'bar': DummyParameter(0, False), 'v': DummyParameter(0, False)}),
                     (True, {'foo': DummyParameter(0, False), 'bar': DummyParameter(0, False), 'v': DummyParameter(0, True)}),
                     (True, {'foo': DummyParameter(0, True), 'bar': DummyParameter(0, True), 'v': DummyParameter(0, True)})]
        for expected_result, parameter_set in test_sets:
            self.assertEqual(expected_result, table.requires_stop(parameter_set))

    def test_identifier(self):
        identifier = 'some name'
        pulse = TablePulseTemplate(identifier=identifier)
        self.assertEqual(pulse.identifier, identifier)


class CleanEntriesTests(unittest.TestCase):

    def test_empty_list(self) -> None:
        self.assertEqual([], clean_entries([]))

    def test_small_list_unchanged(self) -> None:
        table = TablePulseTemplate()
        table.add_entry(0, 5)
        table.add_entry(5, 5)
        clean = clean_entries(table.entries)
        self.assertEqual(clean, table.entries)

    def test_point_removal(self) -> None:
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


class ParameterValueIllegalExceptionTest(unittest.TestCase):

    def test(self) -> None:
        decl = ParameterDeclaration('foo', max=8)
        exception = ParameterValueIllegalException(decl, 8.1)
        self.assertEqual("The value 8.1 provided for parameter foo is illegal (min = -inf, max = 8)", str(exception))

if __name__ == "__main__":
    unittest.main(verbosity=2)
