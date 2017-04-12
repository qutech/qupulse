import unittest
import warnings

import numpy

from qctoolkit.expressions import Expression
from qctoolkit.pulses.instructions import EXECInstruction
from qctoolkit.pulses.table_pulse_template import TablePulseTemplate, TableWaveform, TableEntry, WaveformTableEntry, ZeroDurationTablePulseTemplate
from qctoolkit.pulses.parameters import ParameterDeclaration, ParameterNotProvidedException, ParameterValueIllegalException, ParameterConstraintViolation
from qctoolkit.pulses.interpolation import HoldInterpolationStrategy, LinearInterpolationStrategy, JumpInterpolationStrategy
from qctoolkit.pulses.multi_channel_pulse_template import MultiChannelWaveform

from tests.pulses.sequencing_dummies import DummySequencer, DummyInstructionBlock, DummyInterpolationStrategy, DummyParameter, DummyCondition
from tests.serialization_dummies import DummySerializer


class TableEntryTest(unittest.TestCase):
    def test_known_interpolation_strategies(self):
        strategies = [("linear", LinearInterpolationStrategy()),
                      ("hold", HoldInterpolationStrategy()),
                      ("jump", JumpInterpolationStrategy())]

        for strat_name, strat_val in strategies:
            entry = TableEntry('a', Expression('b'), strat_name)

            self.assertEqual(entry.t, Expression('a'))
            self.assertEqual(entry.v, Expression('b'))
            self.assertEqual(entry.interp, strat_val)

    def test_unknown_interpolation_strategy(self):
        with self.assertRaises(KeyError):
            TableEntry(0, 0, 'foo')


class TablePulseTemplateTest(unittest.TestCase):

    @unittest.skip('Move to AtomicPulseTemplate test')
    def test_measurement_windows(self) -> None:
        pulse = TablePulseTemplate()
        pulse.add_entry(1, 1)
        pulse.add_entry(3, 0)
        pulse.add_entry(5, 0)
        pulse.add_measurement_declaration('mw', 0, 5)
        windows = pulse.get_measurement_windows(parameters={}, measurement_mapping={'mw': 'asd'})
        self.assertEqual([('asd', 0, 5)], windows)
        self.assertEqual(pulse.measurement_declarations, dict(mw=[(0, 5)]))

    @unittest.skip('Move to AtomicPulseTemplate test')
    def test_no_measurement_windows(self) -> None:
        pulse = TablePulseTemplate()
        pulse.add_entry(1, 1)
        pulse.add_entry(3, 0)
        pulse.add_entry(5, 0)
        windows = pulse.get_measurement_windows({}, {'mw': 'asd'})
        self.assertEqual([], windows)
        self.assertEqual(dict(), pulse.measurement_declarations)

    @unittest.skip('Move to AtomicPulseTemplate test')
    def test_measurement_windows_with_parameters(self) -> None:
        pulse = TablePulseTemplate()
        pulse.add_entry(1,        1)
        pulse.add_entry('length', 0)
        pulse.add_measurement_declaration('mw',1,'(1+length)/2')
        parameters = dict(length=100)
        windows = pulse.get_measurement_windows(parameters, measurement_mapping={'mw': 'asd'})
        self.assertEqual(windows, [('asd', 1, 101/2)])
        self.assertEqual(pulse.measurement_declarations, dict(mw=[(1, '(1+length)/2')]))

    @unittest.skip('Move to AtomicPulseTemplate test')
    def test_multiple_measurement_windows(self) -> None:
        pulse = TablePulseTemplate()
        pulse.add_entry(1,        1)
        pulse.add_entry('length', 0)

        pulse.add_measurement_declaration('A', 0, '(1+length)/2')
        pulse.add_measurement_declaration('A', 1, 3)
        pulse.add_measurement_declaration('B', 'begin', 2)

        parameters = dict(length=5, begin=1)
        measurement_mapping = dict(A='A', B='C')
        windows = pulse.get_measurement_windows(parameters=parameters,
                                                measurement_mapping=measurement_mapping)
        expected = [('A', 0, 3), ('A', 1, 3), ('C', 1, 2)]
        self.assertEqual(sorted(windows), sorted(expected))
        self.assertEqual(pulse.measurement_declarations,
                         dict(A=[(0, '(1+length)/2'), (1, 3)],
                              B=[('begin', 2)]))

    def test_time_is_negative(self) -> None:
        with self.assertRaises(ValueError):
            TablePulseTemplate({0: [(1, 2),
                                   (2, 3),
                                   (-1, 3)]})

        with self.assertRaises(ValueError):
            TablePulseTemplate({0: [(-1, 2),
                                   (2, 3),
                                   (3, 3)]})

    def test_time_not_increasing(self):
        with self.assertRaises(ValueError):
            TablePulseTemplate({0: [(1, 2),
                                    (2, 3),
                                    (1.9, 3),
                                    (3, 1.1)]})

        with self.assertRaises(ValueError):
            TablePulseTemplate({0: [('a', 2),
                                    (2, 3),
                                    (1.9, 3),
                                    ('b', 1.1)]})

        with self.assertRaises(ValueError):
            TablePulseTemplate({0: [(2, 3),
                                    ('a', 2),
                                    (1.9, 3),
                                    ('b', 1.1)]})

        with self.assertRaises(ValueError):
            TablePulseTemplate({0: [(1, 2),
                                    (2, 3),
                                    (1.9, 3)]})

        with self.assertRaises(ValueError):
            TablePulseTemplate({0: [(2, 3),
                                    (1.9, 'k')]})

        with self.assertRaises(ValueError):
            TablePulseTemplate({0: [('a', 3),
                                    (2, 'n'),
                                    (3, 'm'),
                                    ('a', 'k')]})

    def test_inconsistent_parameters(self):
        with self.assertRaises(ValueError):
            TablePulseTemplate({0: [('a', 1),
                                    (2, 0)],
                                1: [(3, 6),
                                    ('a', 7)]})

        with self.assertRaises(ValueError):
            TablePulseTemplate({0: [('a', 1),
                                    (2, 0)]}, parameter_constraints=['a>3'])

    @unittest.skip(reason='Needs a better inequality solver')
    def test_time_not_increasing_hard(self):
        with self.assertRaises(ValueError):
            TablePulseTemplate({0: [('a*c', 3),
                                    ('b', 1),
                                    ('c*a', 'k')]}, parameter_constraints=['a*c < b'])



    def test_time_is_0_on_construction(self) -> None:
        with self.assertWarns(ZeroDurationTablePulseTemplate):
            warnings.simplefilter('default', ZeroDurationTablePulseTemplate)
            table = TablePulseTemplate({0: [(0, 1.4)]})
            self.assertTrue(table.duration == 0)
        self.assertTrue(table.duration == 0)

        self.assertIsNone(table.build_waveform(parameters=dict(),
                                               measurement_mapping=dict(),
                                               channel_mapping={0: 0}))

    def test_time_is_0_on_instantiation(self):
        table = TablePulseTemplate({0: [('a', 1)]})
        self.assertEqual(table.duration, Expression('a'))
        self.assertEqual(table.parameter_names, {'a'})

        self.assertIsNone(table.build_waveform(parameters=dict(a=0),
                                               measurement_mapping=dict(),
                                               channel_mapping={0: 0}))

    def test_single_channel_no_parameters(self):
        raw_entries = [(0., 1.1), (1.1, 2.), (2.2, 2.4)]
        table = TablePulseTemplate({0: raw_entries})
        expected = [TableEntry(*entry) for entry in raw_entries]
        self.assertEqual(table.entries, dict([(0, expected)]))
        self.assertEqual(table.duration, 2.2)
        self.assertEqual(table.parameter_names, {})

    def test_single_channel_no_parameters(self):
        raw_entries = [(0., 1.1), (1.1, 2.), (2.2, 2.4)]
        table = TablePulseTemplate({0: raw_entries})
        expected = [TableEntry(*entry) for entry in raw_entries]
        self.assertEqual(table.entries, dict([(0, expected)]))
        self.assertEqual(table.duration, 2.2)
        self.assertEqual(table.parameter_names, {})

    def test_internal_constraints(self):
        table = TablePulseTemplate({0: [(1, 'v'), (2, 'w')],
                                    1: [('t', 'x'), ('t+2', 'y')]},
                                   parameter_constraints=['x<2', 'y<w', 't<1'])
        self.assertEqual(table.parameter_names, {'v', 'w', 't', 'x', 'y'})

        with self.assertRaises(ParameterConstraintViolation):
            table.build_waveform(parameters=dict(v=1, w=2, t=0.1, x=2.2, y=1),
                                 measurement_mapping=dict(),
                                 channel_mapping={0: 0, 1: 1})
        with self.assertRaises(ParameterConstraintViolation):
            table.build_waveform(parameters=dict(v=1, w=2, t=0.1, x=1.2, y=2),
                                 measurement_mapping=dict(),
                                 channel_mapping={0: 0, 1: 1})
        with self.assertRaises(ParameterConstraintViolation):
            table.build_waveform(parameters=dict(v=1, w=2, t=3, x=1.2, y=1),
                                 measurement_mapping=dict(),
                                 channel_mapping={0: 0, 1: 1})
        table.build_waveform(parameters=dict(v=1, w=2, t=0.1, x=1.2, y=1),
                             measurement_mapping=dict(),
                             channel_mapping={0: 0, 1: 1})

    def test_external_constraints(self):
        table = TablePulseTemplate({0: [(1, 'v'), (2, 'w')],
                                    1: [('t', 'x'), ('t+2', 'y')]},
                                   parameter_constraints=['x<h', 'y<w', 't<1'])
        self.assertEqual(table.parameter_names, {'v', 'w', 't', 'x', 'y', 'h'})

        with self.assertRaises(ParameterConstraintViolation):
            table.build_waveform(parameters=dict(v=1., w=2, t=0.1, x=2.2, y=1, h=1),
                                 measurement_mapping=dict(),
                                 channel_mapping={0: 0, 1: 1})
        table.build_waveform(parameters=dict(v=1., w=2, t=0.1, x=1.2, y=1, h=2),
                             measurement_mapping=dict(),
                             channel_mapping={0: 0, 1: 1})

    def test_is_interruptable(self) -> None:
        self.assertFalse(TablePulseTemplate({0: [(1, 1)]}).is_interruptable)

    def test_get_entries_instantiated_one_entry_float_float(self) -> None:
        table = TablePulseTemplate({0: [(0, 2)]})
        instantiated_entries = table.get_entries_instantiated({})[0]
        self.assertEqual([(0, 2, HoldInterpolationStrategy())], instantiated_entries)

    def test_get_entries_instantiated_one_entry_float_declaration(self) -> None:
        table = TablePulseTemplate({0: [(0, 'foo')]})
        instantiated_entries = table.get_entries_instantiated({'foo': 2})[0]
        self.assertEqual([(0, 2, HoldInterpolationStrategy())], instantiated_entries)

    def test_get_entries_instantiated_two_entries_float_float_declaration_float(self) -> None:
        table = TablePulseTemplate({0: [('foo', -2.)]})
        instantiated_entries = table.get_entries_instantiated({'foo': 2})[0]
        self.assertEqual([(0, -2., HoldInterpolationStrategy()),
                          (2, -2., HoldInterpolationStrategy())], instantiated_entries)

    def test_get_entries_instantiated_two_entries_float_declaraton_declaration_declaration(self) -> None:
        table = TablePulseTemplate({0: [(0, 'v1'),
                                        ('t', 'v2')]})
        instantiated_entries = table.get_entries_instantiated({'v1': -5, 'v2': 5, 't': 3})[0]
        self.assertEqual([(0, -5, HoldInterpolationStrategy()),
                          (3, 5, HoldInterpolationStrategy())], instantiated_entries)

    def test_get_entries_instantiated_two_entries_invalid_parameters(self) -> None:
        table = TablePulseTemplate()
        table.add_entry(0, 'v1')
        t_decl = ParameterDeclaration('t', min=1, max=2)
        v2_decl = ParameterDeclaration('v2', min=10, max=30)
        table.add_entry(t_decl, v2_decl)
        with self.assertRaises(ParameterValueIllegalException):
            table.get_entries_instantiated({'v1': -5, 't': 0, 'v2': 20})
        with self.assertRaises(ParameterValueIllegalException):
            table.get_entries_instantiated({'v1': -5, 't': 1, 'v2': -20})

    def test_get_entries_instantiated_two_entries_parameter_missing(self) -> None:
        table = TablePulseTemplate()
        t_decl = ParameterDeclaration('t', min=1, max=2)
        v2_decl = ParameterDeclaration('v2', min=10, max=30)
        table.add_entry(t_decl, v2_decl)
        with self.assertRaises(ParameterNotProvidedException):
            table.get_entries_instantiated(dict())

    def test_get_entries_instantiated_linked_time_declarations(self) -> None:
        table = TablePulseTemplate()
        foo_decl = ParameterDeclaration('foo', min=1)
        bar_decl = ParameterDeclaration('bar')
        table.add_entry(foo_decl, 'v', 'linear')
        table.add_entry(bar_decl, 0, 'jump')
        instantiated_entries = table.get_entries_instantiated({'v': 2.3, 'foo': 1, 'bar': 4})['default']
        self.assertEqual([(0, 0, HoldInterpolationStrategy()), (1, 2.3, LinearInterpolationStrategy()), (4, 0, JumpInterpolationStrategy())], instantiated_entries)
        with self.assertRaises(Exception):
            table.get_entries_instantiated({'v': 2.3, 'foo': 1, 'bar': 1})
        with self.assertRaises(Exception):
            table.get_entries_instantiated({'v': 2.3, 'foo': 1, 'bar': 0.2})

    def test_get_entries_instantiated_unlinked_time_declarations(self) -> None:
        table = TablePulseTemplate()
        foo_decl = ParameterDeclaration('foo', min=1, max=2)
        bar_decl = ParameterDeclaration('bar', min=1.5, max=4)
        table.add_entry(foo_decl, 'v', 'linear')
        table.add_entry(bar_decl, 0, 'jump')
        instantiated_entries = table.get_entries_instantiated({'v': 2.3, 'foo': 1, 'bar': 4})['default']
        self.assertEqual([(0, 0, HoldInterpolationStrategy()), (1, 2.3, LinearInterpolationStrategy()), (4, 0, JumpInterpolationStrategy())], instantiated_entries)
        with self.assertRaises(Exception):
            table.get_entries_instantiated({'v': 2.3, 'foo': 2, 'bar': 1.5})

    def test_get_entries_instantiated_empty(self) -> None:
        table = TablePulseTemplate()
        self.assertEquals([(0, 0, HoldInterpolationStrategy())], table.get_entries_instantiated({})['default'])

    def test_get_entries_instantiated_two_equal_entries(self) -> None:
        table = TablePulseTemplate()
        table.add_entry(0, 0)
        table.add_entry(1, 5)
        table.add_entry(3, 5)
        table.add_entry(5, 1)
        entries = table.get_entries_instantiated({})['default']
        expected = [
            TableEntry(0, 0, HoldInterpolationStrategy()),
            TableEntry(1, 5, HoldInterpolationStrategy()),
            TableEntry(3, 5, HoldInterpolationStrategy()),
            TableEntry(5, 1, HoldInterpolationStrategy())
        ]
        self.assertEqual(expected, entries)

    def test_get_entries_instantiated_removal_for_three_subsequent_equal_entries(self) -> None:
        table = TablePulseTemplate()
        table.add_entry(1, 5)
        table.add_entry(1.5, 5)
        table.add_entry(2, 5)
        table.add_entry(3, 0)
        entries = table.get_entries_instantiated({})['default']
        expected = [
            TableEntry(0, 0, HoldInterpolationStrategy()),
            TableEntry(1, 5, HoldInterpolationStrategy()),
            TableEntry(2, 5, HoldInterpolationStrategy()),
            TableEntry(3, 0, HoldInterpolationStrategy())
        ]
        self.assertEqual(expected, entries)

    def test_get_entries_instantiated_removal_for_three_subsequent_equal_entries_does_not_destroy_linear_interpolation(self) -> None:
        table = TablePulseTemplate()
        table.add_entry(0, 5)
        table.add_entry(2, 5, 'linear')
        table.add_entry(5, 5)
        table.add_entry(10, 0, 'linear')

        entries = table.get_entries_instantiated({})['default']

        expected = [
            TableEntry(0, 5, HoldInterpolationStrategy()),
            TableEntry(5, 5, HoldInterpolationStrategy()),
            TableEntry(10, 0, LinearInterpolationStrategy())
        ]
        self.assertEqual(expected, entries)

        result_sampled = TableWaveform(channel='A', waveform_table=entries, measurement_windows=[]).get_sampled(
            channel='A',
            sample_times=numpy.linspace(0, 10, 11))

        numbers = [5, 5, 5, 5, 5, 5, 4, 3, 2, 1, 0]
        expected = [float(x) for x in numbers]
        self.assertEqual(expected, result_sampled.tolist())

    def test_get_entries_instantiated_two_channels_one_empty(self) -> None:
        table = TablePulseTemplate(channels=['A','B'])
        table.add_entry('foo', 4, channel='A')
        parameters = {'foo': 10}

        entries = table.get_entries_instantiated(parameters)

        expected = {
            'A': [
                TableEntry(0, 0, HoldInterpolationStrategy()),
                TableEntry(10, 4, HoldInterpolationStrategy()),
            ],
            'B': [
                TableEntry(0, 0, HoldInterpolationStrategy()),
                TableEntry(10, 0, HoldInterpolationStrategy())
            ]
        }

        self.assertEqual(expected, entries)

    def test_from_array_1D(self) -> None:
        times = numpy.array([0, 1, 3])
        voltages = numpy.array([5, 0, 5])
        pulse = TablePulseTemplate.from_array(times, voltages)
        entries = []
        for (time, voltage) in zip(times, voltages):
            entries.append(TableEntry(time, voltage, HoldInterpolationStrategy()))
        self.assertEqual(entries, pulse.entries)

    def test_from_array_multi(self) -> None:
        times = numpy.array([0, 1, 3])
        voltages = numpy.array([[1,2,3],
                                [2,3,4]]).T # todo: why transposed??
        pulse = TablePulseTemplate.from_array(times, voltages, [0, 1])
        entries = {
            i: [TableEntry(time, voltage, HoldInterpolationStrategy())
                for (time, voltage) in zip(times, channel_voltage)]
        for i, channel_voltage in enumerate(voltages.T)}

        self.assertEqual(entries, pulse.entries)

    def test_add_entry_multi_invalid_channel(self) -> None:
        pulse = TablePulseTemplate()
        with self.assertRaises(ValueError):
            pulse.add_entry(2,2, channel=1)

    def test_add_entry_multi(self) -> None:
        pulse = TablePulseTemplate(channels=[0, 1])
        pulse.add_entry(1,1, channel=0)
        pulse.add_entry(1,1, channel=1)
        entries = {0: [(0,0,HoldInterpolationStrategy()),
                    (1,1,HoldInterpolationStrategy())],
                   1: [(0,0,HoldInterpolationStrategy()),
                    (1,1,HoldInterpolationStrategy())]}
        self.assertEqual(entries, pulse.entries)

    def test_add_entry_multi_same_time_param(self) -> None:
        pulse = TablePulseTemplate(channels=[0, 1])
        pulse.add_entry(1, 3, channel=0)
        pulse.add_entry('foo', 'bar', channel=0)
        pulse.add_entry(7, 3, channel=0)

        pulse.add_entry(0, -5, channel=1)
        pulse.add_entry(0.5, -2, channel=1)
        pulse.add_entry('foo', 0, channel=1)
        pulse.add_entry(5, 'bar', channel=1)

        expected_foo = ParameterDeclaration('foo', min=1, max=5)
        expected_bar = ParameterDeclaration('bar')
        entries = {0: [TableEntry(0, 0, HoldInterpolationStrategy()),
                    TableEntry(1, 3, HoldInterpolationStrategy()),
                    TableEntry(expected_foo, expected_bar, HoldInterpolationStrategy()),
                    TableEntry(7, 3, HoldInterpolationStrategy())],
                   1: [TableEntry(0, -5, HoldInterpolationStrategy()),
                    TableEntry(0.5, -2, HoldInterpolationStrategy()),
                    TableEntry(expected_foo, 0, HoldInterpolationStrategy()),
                    TableEntry(5, expected_bar, HoldInterpolationStrategy())]}
        self.assertEqual(entries, pulse.entries)
        self.assertEqual({'foo', 'bar'}, pulse.parameter_names)
        self.assertEqual({expected_bar, expected_foo}, pulse.parameter_declarations)

    def test_get_instantiated_entries_multi_same_time_param(self) -> None:
        table = TablePulseTemplate(channels=[0, 1])
        table.add_entry(1, 3, channel=0)
        table.add_entry('foo', 'bar', channel=0)
        table.add_entry(7, 3, channel=0)

        table.add_entry(0, -5, channel=1)
        table.add_entry(0.5, -2, channel=1)
        table.add_entry('foo', 0, channel=1)
        table.add_entry(5, 'bar', channel=1)

        parameters = {'foo': 2.7, 'bar': -3.3}

        entries = table.get_entries_instantiated(parameters)

        expected = {
            0: [
                TableEntry(0, 0, HoldInterpolationStrategy()),
                TableEntry(1, 3, HoldInterpolationStrategy()),
                TableEntry(2.7, -3.3, HoldInterpolationStrategy()),
                TableEntry(7, 3, HoldInterpolationStrategy()),
            ],
            1: [
                TableEntry(0, -5, HoldInterpolationStrategy()),
                TableEntry(0.5, -2, HoldInterpolationStrategy()),
                TableEntry(2.7, 0, HoldInterpolationStrategy()),
                TableEntry(5, -3.3, HoldInterpolationStrategy()),
                TableEntry(7, -3.3, HoldInterpolationStrategy())
            ]
        }

        self.assertEqual(expected, entries)

    def test_get_instaniated_entries_multi_one_empty_channel(self) -> None:
        table = TablePulseTemplate(channels=[0, 1])
        table.add_entry(1, 3, channel=1)
        table.add_entry('foo', 'bar', 'linear', channel=1)

        parameters = {'foo': 5.2, 'bar': -83.8}

        entries = table.get_entries_instantiated(parameters)

        expected = {
            0: [
                TableEntry(0, 0, HoldInterpolationStrategy()),
                TableEntry(5.2, 0, HoldInterpolationStrategy())
            ],
            1: [
                TableEntry(0, 0, HoldInterpolationStrategy()),
                TableEntry(1, 3, HoldInterpolationStrategy()),
                TableEntry(5.2, -83.8, LinearInterpolationStrategy())
            ]
        }

        self.assertEqual(expected, entries)

    def test_measurement_windows_multi(self) -> None:
        pulse = TablePulseTemplate(channels=[0, 1])
        pulse.add_entry(1, 1, channel=0)
        pulse.add_entry(3, 0, channel=0)
        pulse.add_entry(5, 0, channel=0)

        pulse.add_entry(1, 1, channel=1)
        pulse.add_entry(3, 0, channel=1)
        pulse.add_entry(10, 0, channel=1)

        pulse.add_measurement_declaration('mw',1,7)
        windows = pulse.get_measurement_windows({}, measurement_mapping={'mw': 'asd'})
        self.assertEqual([('asd',1,7)], windows)

    def test_measurement_windows_multi_out_of_pulse(self) -> None:
        pulse = TablePulseTemplate(channels=[0, 1])
        pulse.add_entry(1, 1, channel=0)
        pulse.add_entry(3, 0, channel=0)
        pulse.add_entry(5, 0, channel=0)

        pulse.add_entry(1, 1, channel=1)
        pulse.add_entry(3, 0, channel=1)
        pulse.add_entry(10, 0, channel=1)

        with self.assertRaises(ValueError):
            pulse.add_measurement_declaration('mw', 1, 't_meas')
            pulse.get_measurement_windows({'t_meas': 20}, measurement_mapping={'mw': 'asd'})


@unittest.skip
class TablePulseTemplateSerializationTests(unittest.TestCase):

    def setUp(self) -> None:
        self.serializer = DummySerializer(lambda x: dict(name=x.name), lambda x: x.name, lambda x: x['name'])
        self.template = TablePulseTemplate(identifier='foo', channels=['A', 'B'])
        self.expected_data = dict(type=self.serializer.get_type_identifier(self.template))
        self.maxDiff = None

    def test_get_serialization_data(self) -> None:
        self.template.add_entry('foo', 2, channel='A')
        self.template.add_entry('hugo', 'ilse', interpolation='linear',channel='A')

        self.template.add_entry(2, 2, channel='B', interpolation='jump')

        self.template.add_measurement_declaration('mw',2,'hugo+franz')

        self.expected_data['measurement_declarations'] = {'mw': [(2,'hugo+franz')]}
        self.expected_data['time_parameter_declarations'] = [dict(name=name) for name in sorted(['foo','hugo','franz'])]
        self.expected_data['voltage_parameter_declarations'] = [dict(name='ilse')]
        self.expected_data['entries'] = dict(A=[(0, 0, 'hold'), ('foo', 2, 'hold'), ('hugo', 'ilse', 'linear')], B=[(0, 0, 'hold'), (2, 2, 'jump')])

        data = self.template.get_serialization_data(self.serializer)
        self.assertEqual(self.expected_data, data)

    def test_deserialize(self) -> None:
        data = dict(measurement_declarations={'mw': [(2,'hugo+franz')]},
                    time_parameter_declarations=[dict(name='hugo'), dict(name='foo'), dict(name='franz')],
                    voltage_parameter_declarations=[dict(name='ilse')],
                    entries=dict(default=[(0, 0, 'hold'), ('foo', 2, 'hold'), ('hugo', 'ilse', 'linear')]),
                    identifier='foo')

        # prepare dependencies for deserialization
        self.serializer.subelements['foo'] = ParameterDeclaration('foo')
        self.serializer.subelements['hugo'] = ParameterDeclaration('hugo')
        self.serializer.subelements['ilse'] = ParameterDeclaration('ilse')
        self.serializer.subelements['franz'] = ParameterDeclaration('franz')

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
        self.assertEqual({'foo', 'hugo', 'ilse', 'franz'}, template.parameter_names)
        self.assertEqual(entries, template.entries)
        self.assertEqual('foo', template.identifier)


@unittest.skip
class TablePulseTemplateSequencingTests(unittest.TestCase):

    def test_build_sequence(self) -> None:
        table = TablePulseTemplate()
        foo_decl = ParameterDeclaration('foo', min=1)
        bar_decl = ParameterDeclaration('bar')
        table.add_entry(foo_decl, 'v', 'linear')
        table.add_entry(bar_decl, 0, 'jump')
        parameters = {'v': 2.3, 'foo': 1, 'bar': 4}
        instantiated_entries = table.get_entries_instantiated(parameters)
        channel_mapping = {'default': 'default'}
        waveform = table.build_waveform(parameters,
                                        measurement_mapping={},
                                        channel_mapping=channel_mapping)
        sequencer = DummySequencer()
        instruction_block = DummyInstructionBlock()

        table.build_sequence(sequencer, parameters, {}, {}, channel_mapping, instruction_block)
        if len(instantiated_entries) == 1:
            expected_waveform = TableWaveform(*instantiated_entries.popitem(), measurement_windows=[])
        else:
            expected_waveform = MultiChannelWaveform([TableWaveform(channel=channel,
                                                                    waveform_table=inst,
                                                                    measurement_windows=[])
                                                      for channel, inst in instantiated_entries.items()])
        self.assertEqual(1, len(instruction_block.instructions))
        instruction = instruction_block.instructions[0]
        self.assertIsInstance(instruction, EXECInstruction)
        self.assertEqual(expected_waveform, instruction.waveform)
        self.assertEqual(expected_waveform, waveform)

    @unittest.skip("What exactly is the point of allowing empty/non-existent waveforms?")
    def test_build_sequence_empty(self) -> None:
        table = TablePulseTemplate()
        sequencer = DummySequencer()
        instruction_block = DummyInstructionBlock()
        table.build_sequence(sequencer, {}, {}, {}, instruction_block)
        self.assertFalse(instruction_block.instructions)
        self.assertIsNone(table.build_waveform({}))

    def test_requires_stop_missing_param(self) -> None:
        table = TablePulseTemplate()
        foo_decl = ParameterDeclaration('foo')
        table.add_entry(foo_decl, 'v', 'linear')
        with self.assertRaises(ParameterNotProvidedException):
            table.requires_stop({'foo': DummyParameter(0, False)}, {})

    def test_requires_stop(self) -> None:
        table = TablePulseTemplate()
        foo_decl = ParameterDeclaration('foo', min=1)
        bar_decl = ParameterDeclaration('bar')
        table.add_entry(foo_decl, 'v', 'linear')
        table.add_entry(bar_decl, 0, 'jump')
        test_sets = [(False, {'foo': DummyParameter(0, False), 'bar': DummyParameter(0, False), 'v': DummyParameter(0, False)}, {'foo': DummyCondition(False)}),
                     (False, {'foo': DummyParameter(0, False), 'bar': DummyParameter(0, False), 'v': DummyParameter(0, False)}, {'foo': DummyCondition(True)}),
                     (True, {'foo': DummyParameter(0, True), 'bar': DummyParameter(0, False), 'v': DummyParameter(0, False)}, {'foo': DummyCondition(False)}),
                     (True, {'foo': DummyParameter(0, True), 'bar': DummyParameter(0, False), 'v': DummyParameter(0, False)}, {'foo': DummyCondition(True)}),
                     (True, {'foo': DummyParameter(0, False), 'bar': DummyParameter(0, False), 'v': DummyParameter(0, True)}, {'foo': DummyCondition(False)}),
                     (True, {'foo': DummyParameter(0, False), 'bar': DummyParameter(0, False), 'v': DummyParameter(0, True)}, {'foo': DummyCondition(True)}),
                     (True, {'foo': DummyParameter(0, True), 'bar': DummyParameter(0, True), 'v': DummyParameter(0, True)}, {'foo': DummyCondition(False)}),
                     (True, {'foo': DummyParameter(0, True), 'bar': DummyParameter(0, True), 'v': DummyParameter(0, True)}, {'foo': DummyCondition(True)})]
        for expected_result, parameter_set, condition_set in test_sets:
            self.assertEqual(expected_result, table.requires_stop(parameter_set, condition_set))

    def test_identifier(self) -> None:
        identifier = 'some name'
        pulse = TablePulseTemplate(identifier=identifier)
        self.assertEqual(pulse.identifier, identifier)

    def test_build_sequence_multi(self) -> None:
        table = TablePulseTemplate(channels=['A', 'B'])
        table.add_entry(1, 3, channel='A')
        table.add_entry('foo', 'bar', channel='A')
        table.add_entry(7, 3, channel='A')

        table.add_entry(0, -5, channel='B')
        table.add_entry('foo', 0, channel='B')
        table.add_entry(5, 'bar', channel='B')

        parameters = {'foo': 3, 'bar': 17}
        channel_mapping = {'A': 'CHA', 'B': 'CHB'}

        instantiated_entries = table.get_entries_instantiated(parameters)
        expected_waveform = MultiChannelWaveform(
            [TableWaveform(channel=('CH'+channel), waveform_table=instantiated, measurement_windows=[])
             for channel, instantiated in
             instantiated_entries.items()])

        sequencer = DummySequencer()
        instruction_block = DummyInstructionBlock()
        table.build_sequence(sequencer, parameters, {}, {},
                             channel_mapping=channel_mapping,
                             instruction_block=instruction_block)

        self.assertEqual(1, len(instruction_block.instructions))
        instruction = instruction_block.instructions[0]
        self.assertIsInstance(instruction, EXECInstruction)
        self.assertEqual(expected_waveform, instruction.waveform)
        waveform = table.build_waveform(parameters, measurement_mapping={}, channel_mapping=channel_mapping)
        for ch in waveform.defined_channels:
            self.assertEqual(expected_waveform, waveform)

    def test_build_sequence_multi_one_channel_empty(self) -> None:
        table = TablePulseTemplate(channels={'A', 'B'})
        table.add_entry('foo', 4, channel='A')
        parameters = {'foo': 3}
        channel_mapping = {'A': 'CHA', 'B': 'CHB'}

        instantiated_entries = table.get_entries_instantiated(parameters)

        sequencer = DummySequencer()
        instruction_block = DummyInstructionBlock()
        table.build_sequence(sequencer, parameters,
                             conditions={},
                             measurement_mapping={},
                             channel_mapping=channel_mapping,
                             instruction_block=instruction_block)
        expected_waveform = MultiChannelWaveform([TableWaveform('CH'+channel, instantiated, []) for channel, instantiated in instantiated_entries.items()])
        self.assertEqual(1, len(instruction_block.instructions))
        instruction = instruction_block.instructions[0]
        self.assertIsInstance(instruction, EXECInstruction)
        self.assertEqual(expected_waveform, instruction.waveform)
        waveform = table.build_waveform(parameters, measurement_mapping={}, channel_mapping=channel_mapping)

        self.assertEqual(expected_waveform, waveform)


@unittest.skip
class TableWaveformDataTests(unittest.TestCase):

    def test_duration(self) -> None:
        entries = [WaveformTableEntry(0, 0, HoldInterpolationStrategy()), WaveformTableEntry(5, 1, HoldInterpolationStrategy())]
        waveform = TableWaveform('A', entries, [])
        self.assertEqual(5, waveform.duration)

    @unittest.skip("What is the point of empty waveforms?")
    def test_duration_no_entries(self) -> None:
        waveform = TableWaveform([])
        self.assertEqual(0, waveform.duration)

    def test_duration_no_entries_exception(self) -> None:
        with self.assertRaises(ValueError):
            waveform = TableWaveform('A', [], [])
            self.assertEqual(0, waveform.duration)

    def test_few_entries(self) -> None:
        with self.assertRaises(ValueError):
            TableWaveform('A', [[]], [])
        with self.assertRaises(ValueError):
            TableWaveform('A', [WaveformTableEntry(0, 0, HoldInterpolationStrategy())], [])

    def test_unsafe_sample(self) -> None:
        interp = DummyInterpolationStrategy()
        entries = [WaveformTableEntry(0, 0, interp),
                   WaveformTableEntry(2.1, -33.2, interp),
                   WaveformTableEntry(5.7, 123.4, interp)]
        waveform = TableWaveform('A', entries, [])
        sample_times = numpy.linspace(.5, 5.5, num=11)

        expected_interp_arguments = [((0, 0), (2.1, -33.2), [0.5, 1.0, 1.5, 2.0]),
                                     ((2.1, -33.2), (5.7, 123.4), [2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5])]
        expected_result = numpy.copy(sample_times)

        result = waveform.unsafe_sample('A', sample_times)

        self.assertEqual(expected_interp_arguments, interp.call_arguments)
        numpy.testing.assert_equal(expected_result, result)

        output_expected = numpy.empty_like(expected_result)
        output_received = waveform.unsafe_sample('A', sample_times, output_array=output_expected)
        self.assertIs(output_expected, output_received)
        numpy.testing.assert_equal(expected_result, output_received)

    def test_simple_properties(self):
        interp = DummyInterpolationStrategy()
        entries = [WaveformTableEntry(0, 0, interp),
                   WaveformTableEntry(2.1, -33.2, interp),
                   WaveformTableEntry(5.7, 123.4, interp)]
        meas = [('M', 1, 2)]
        chan = 'A'
        waveform = TableWaveform(chan, entries, meas)

        self.assertEqual(waveform.defined_channels, {chan})
        self.assertEqual(list(waveform.get_measurement_windows()), meas)
        self.assertIs(waveform.unsafe_get_subset_for_channels('A'), waveform)


@unittest.skip
class ParameterValueIllegalExceptionTest(unittest.TestCase):

    def test(self) -> None:
        decl = ParameterDeclaration('foo', max=8)
        exception = ParameterValueIllegalException(decl, 8.1)
        self.assertEqual("The value 8.1 provided for parameter foo is illegal (min = -inf, max = 8)", str(exception))

if __name__ == "__main__":
    unittest.main(verbosity=2)
