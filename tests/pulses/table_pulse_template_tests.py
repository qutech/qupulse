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
        self.assertEqual(table.parameter_names, set())

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
                                 measurement_mapping={0: 0, 1: 1},
                                 channel_mapping={0: 0, 1: 1})
        table.build_waveform(parameters=dict(v=1., w=2, t=0.1, x=1.2, y=1, h=2),
                             measurement_mapping={0: 0, 1: 1},
                             channel_mapping={0: 0, 1: 1})

    def test_is_interruptable(self) -> None:
        self.assertFalse(TablePulseTemplate({0: [(1, 1)]}).is_interruptable)

    def test_get_entries_instantiated_one_entry_float_float(self) -> None:
        table = TablePulseTemplate({0: [(0, 2)]})
        instantiated_entries = table.get_entries_instantiated(dict())[0]
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

    def test_get_entries_instantiated_multiple_parameters_missing(self) -> None:
        table = TablePulseTemplate({0: [(0, 'v1'),
                                        ('t', 'v2')]})
        with self.assertRaises(ParameterNotProvidedException):
            table.get_entries_instantiated(dict())
        with self.assertRaises(ParameterNotProvidedException):
            table.get_entries_instantiated(dict(v1=1))
        with self.assertRaises(ParameterNotProvidedException):
            table.get_entries_instantiated(dict(v1=1, t=2))
        table.get_entries_instantiated(dict(v1=1, t=2, v2=2))

    def test_get_entries_auto_insert(self) -> None:
        table = TablePulseTemplate({0: [('foo', 'v', 'linear'),
                                        ('bar', 0, 'jump')],
                                    1: [(0, 3, 'linear'),
                                        ('bar+foo', 2, 'linear')]})
        instantiated_entries = table.get_entries_instantiated({'v': 2.3, 'foo': 1, 'bar': 4})
        self.assertEqual({0: [(0, 2.3, HoldInterpolationStrategy()),
                              (1, 2.3, LinearInterpolationStrategy()),
                              (4, 0, JumpInterpolationStrategy()),
                              (5, 0, HoldInterpolationStrategy())],
                          1: [(0, 3, LinearInterpolationStrategy()),
                              (5, 2, LinearInterpolationStrategy())]}, instantiated_entries)

    def test_empty_instantiated(self) -> None:
        with self.assertRaises(TypeError):
            TablePulseTemplate()
        with self.assertRaises(ValueError):
            TablePulseTemplate(entries=dict())

    def test_get_entries_instantiated_two_equal_entries(self) -> None:
        table = TablePulseTemplate({0: [(0, 0),
                                        (1, 5),
                                        (3, 5),
                                        (5, 1)]})
        entries = table.get_entries_instantiated(dict())
        expected = [
            TableEntry(0, 0, HoldInterpolationStrategy()),
            TableEntry(1, 5, HoldInterpolationStrategy()),
            TableEntry(3, 5, HoldInterpolationStrategy()),
            TableEntry(5, 1, HoldInterpolationStrategy())
        ]
        self.assertEqual({0: expected}, entries)

    def test_get_entries_instantiated_removal_for_three_subsequent_equal_entries(self) -> None:
        table = TablePulseTemplate({0: [(1, 5),
                                        (1.5, 5),
                                        (2, 5),
                                        (3, 0)]})
        entries = table.get_entries_instantiated(dict())
        expected = [
            TableEntry(0, 5, HoldInterpolationStrategy()),
            TableEntry(2, 5, HoldInterpolationStrategy()),
            TableEntry(3, 0, HoldInterpolationStrategy())
        ]
        self.assertEqual({0: expected}, entries)

    def test_get_entries_instantiated_removal_for_three_subsequent_equal_entries_does_not_destroy_linear_interpolation(self) -> None:
        table = TablePulseTemplate({0: [(0, 5),
                                        (2, 5, 'linear'),
                                        (5, 5),
                                        (10, 0, 'linear')]})
        entries = table.get_entries_instantiated(dict())

        expected = [
            TableEntry(0, 5, HoldInterpolationStrategy()),
            TableEntry(5, 5, HoldInterpolationStrategy()),
            TableEntry(10, 0, LinearInterpolationStrategy())
        ]
        self.assertEqual({0: expected}, entries)

    def test_from_array_exceptions(self):
        with self.assertRaises(ValueError):
            TablePulseTemplate.from_array(numpy.arange(0), numpy.arange(1), [0])

        with self.assertRaises(ValueError):
            TablePulseTemplate.from_array(numpy.arange(1), numpy.arange(0), [0])

        with self.assertRaises(ValueError):
            TablePulseTemplate.from_array(numpy.array(numpy.ndindex((1, 2, 1))), numpy.arange(2), [0])

        with self.assertRaises(ValueError):
            TablePulseTemplate.from_array(numpy.array(numpy.ndindex((3, 4))),
                                          numpy.array(numpy.ndindex((3, 5))), [3, 4, 5])

        with self.assertRaises(ValueError):
            TablePulseTemplate.from_array(numpy.array(numpy.ndindex((3, 5))),
                                          numpy.array(numpy.ndindex((3, 4))), [3, 4, 5])

        with self.assertRaises(ValueError):
            TablePulseTemplate.from_array(numpy.array(numpy.ndindex((2, 4))),
                                          numpy.array(numpy.ndindex((3, 4))), [3, 4, 5])

        with self.assertRaises(ValueError):
            TablePulseTemplate.from_array(numpy.array(numpy.ndindex((3, 4))),
                                          numpy.array(numpy.ndindex((2, 4))), [3, 4, 5])

    def test_from_array_1D(self) -> None:
        times = numpy.array([0, 1, 3])
        voltages = numpy.array([5, 0, 5])
        pulse = TablePulseTemplate.from_array(times, voltages, [0])
        entries = []
        for (time, voltage) in zip(times, voltages):
            entries.append(TableEntry(time, voltage, HoldInterpolationStrategy()))
        self.assertEqual({0: entries}, pulse.entries)

    def test_from_array_multi_one_time(self) -> None:
        times = numpy.array([0, 1, 3])
        voltages = numpy.array([[1, 2, 3],
                                [2, 3, 4]])
        pulse = TablePulseTemplate.from_array(times, voltages, [0, 1])
        entries = {
            i: [TableEntry(time, voltage, HoldInterpolationStrategy())
                for (time, voltage) in zip(times, voltages[i, :])]
            for i in range(2)}

        self.assertEqual(entries, pulse.entries)

    def test_from_array_multi(self) -> None:
        times = numpy.array([[0, 1, 3], [2, 3, 4]])
        voltages = numpy.array([[1, 2, 3],
                                [2, 3, 4]])
        pulse = TablePulseTemplate.from_array(times, voltages, [0, 1])
        entries = {
            i: [TableEntry(time, voltage, HoldInterpolationStrategy())
                for (time, voltage) in zip(times[i, :], voltages[i, :])]
            for i in range(2)}

        self.assertEqual(entries, pulse.entries)

    def test_from_array_multi_one_voltage(self) -> None:
        times = numpy.array([[0, 1, 3], [2, 3, 4]])
        voltages = numpy.array([1, 2, 3])
        pulse = TablePulseTemplate.from_array(times, voltages, [0, 1])
        entries = {
            i: [TableEntry(time, voltage, HoldInterpolationStrategy())
                for (time, voltage) in zip(times[i, :], voltages)]
            for i in range(2)}

        self.assertEqual(entries, pulse.entries)

    def test_add_entry_multi_same_time_param(self) -> None:
        pulse = TablePulseTemplate({0: [(1, 3),
                                        ('foo', 'bar'),
                                        (7, 3)],
                                    1: [(0, -5),
                                        (0.5, -2),
                                        ('foo', 0),
                                        (5, 'bar')]})
        self.assertEqual({'foo', 'bar'}, pulse.parameter_names)

    def test_get_instantiated_entries_multi_same_time_param(self) -> None:
        table = TablePulseTemplate({0: [(1, 3),
                                        ('foo', 'bar'),
                                        (7, 3)],
                                    1: [(0, -5),
                                        (0.5, -2),
                                        ('foo', 0),
                                        (5, 'bar')]})
        parameters = {'foo': 2.7, 'bar': -3.3}

        entries = table.get_entries_instantiated(parameters)

        expected = {
            0: [
                TableEntry(0, 3, HoldInterpolationStrategy()),
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


class TablePulseTemplateSerializationTests(unittest.TestCase):

    def setUp(self) -> None:
        self.serializer = DummySerializer(lambda x: dict(name=x.name), lambda x: x.name, lambda x: x['name'])
        self.entries = dict(A=[('foo', 2, 'hold'), ('hugo', 'ilse', 'linear')],
                            B=[(0, 5, 'hold'), (1, 7, 'jump'), ('k', 't', 'hold')])
        self.measurements = [('m', 1, 1), ('foo', 'z', 'o')]
        self.template = TablePulseTemplate(entries=self.entries,
                                           measurements=self.measurements,
                                           identifier='foo', parameter_constraints=['ilse>2', 'k>foo'])
        self.expected_data = dict(type=self.serializer.get_type_identifier(self.template))
        self.maxDiff = None

    def test_get_serialization_data(self) -> None:
        expected_data = dict(measurements=self.measurements,
                             entries=self.entries,
                             parameter_constraints=[str(Expression('ilse>2')), str(Expression('k>foo'))])

        data = self.template.get_serialization_data(self.serializer)
        self.assertEqual(expected_data, data)

    def test_deserialize(self) -> None:
        data = dict(measurements=self.measurements,
                    entries=self.entries,
                    parameter_constraints=['ilse>2', 'k>foo'],
                    identifier='foo')

        # deserialize
        template = TablePulseTemplate.deserialize(self.serializer, **data)

        self.assertEqual(template.entries, self.template.entries)
        self.assertEqual(template.measurement_declarations, self.template.measurement_declarations)


class TablePulseTemplateSequencingTests(unittest.TestCase):
    def test_build_waveform_single_channel(self):
        table = TablePulseTemplate({0: [(0, 0),
                                        ('foo', 'v', 'linear'),
                                        ('bar', 0, 'jump')]},
                                   parameter_constraints=['foo>1'],
                                   measurements=[('M', 'b', 'l'),
                                                 ('N', 1, 2)])

        parameters = {'v': 2.3, 'foo': 1, 'bar': 4, 'b': 2, 'l': 1}
        measurement_mapping = {'M': 'P', 'N': 'N'}
        channel_mapping = {0: 'ch'}

        with self.assertRaises(ParameterConstraintViolation):
            table.build_waveform(parameters=parameters,
                                 measurement_mapping=measurement_mapping,
                                 channel_mapping=channel_mapping)

        parameters['foo'] = 1.1
        waveform = table.build_waveform(parameters=parameters,
                                        measurement_mapping=measurement_mapping,
                                        channel_mapping=channel_mapping)

        self.assertIsInstance(waveform, TableWaveform)
        self.assertEqual(waveform._table,
                         ((0, 0, HoldInterpolationStrategy()),
                          (1.1, 2.3, LinearInterpolationStrategy()),
                          (4, 0, JumpInterpolationStrategy())))
        self.assertEqual(sorted(waveform._measurement_windows),
                         sorted((('P', 2, 1),
                                 ('N', 1, 2))))
        self.assertEqual(waveform._channel_id,
                         'ch')

    def test_build_waveform_multi_channel(self):
        table = TablePulseTemplate({0: [(0, 0),
                                        ('foo', 'v', 'linear'),
                                        ('bar', 0, 'jump')],
                                    3: [(0, 1),
                                        ('bar+foo', 0, 'linear')]},
                                   parameter_constraints=['foo>1'],
                                   measurements=[('M', 'b', 'l'),
                                                 ('N', 1, 2)])

        parameters = {'v': 2.3, 'foo': 1, 'bar': 4, 'b': 2, 'l': 1}
        measurement_mapping = {'M': 'P', 'N': 'N'}
        channel_mapping = {0: 'ch', 3: 'oh'}

        with self.assertRaises(ParameterConstraintViolation):
            table.build_waveform(parameters=parameters,
                                 measurement_mapping=measurement_mapping,
                                 channel_mapping=channel_mapping)

        parameters['foo'] = 1.1
        waveform = table.build_waveform(parameters=parameters,
                                        measurement_mapping=measurement_mapping,
                                        channel_mapping=channel_mapping)

        self.assertIsInstance(waveform, MultiChannelWaveform)
        self.assertEqual(len(waveform._sub_waveforms), 2)

        channels = {'oh', 'ch'}
        found_measurements = False
        for wf in waveform._sub_waveforms:
            self.assertIsInstance(wf, TableWaveform)
            self.assertIn(wf._channel_id, channels)
            channels.remove(wf._channel_id)
            if wf.defined_channels == {'ch'}:
                self.assertEqual(wf._table,
                                 ((0, 0, HoldInterpolationStrategy()),
                                  (1.1, 2.3, LinearInterpolationStrategy()),
                                  (4, 0, JumpInterpolationStrategy()),
                                  (5.1, 0, HoldInterpolationStrategy())))
            elif wf.defined_channels == {'oh'}:
                self.assertEqual(wf._table,
                                 ((0, 1, HoldInterpolationStrategy()),
                                  (5.1, 0, LinearInterpolationStrategy())))

            if wf._measurement_windows:
                self.assertFalse(found_measurements, 'Measurements found twice')
                self.assertEqual(sorted(wf._measurement_windows),
                                 sorted((('P', 2, 1),
                                         ('N', 1, 2))))
                found_measurements = True
        self.assertTrue(found_measurements, 'Measurements not found')

    def test_build_waveform_empty(self) -> None:
        table = TablePulseTemplate(dict(a=[('t', 0)]))
        self.assertIsNone(table.build_waveform(dict(t=0), dict(), dict(a='a')))

    def test_requires_stop_missing_param(self) -> None:
        table = TablePulseTemplate({0: [('foo', 'v')]})
        with self.assertRaises(ParameterNotProvidedException):
            table.requires_stop({'foo': DummyParameter(0, False)}, {})

    def test_requires_stop(self) -> None:
        table = TablePulseTemplate({0: [('foo', 'v'),
                                        ('bar', 0)]})
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
        pulse = TablePulseTemplate(entries={0: [(1, 0)]}, identifier=identifier)
        self.assertEqual(pulse.identifier, identifier)



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


class ParameterValueIllegalExceptionTest(unittest.TestCase):

    def test(self) -> None:
        decl = ParameterDeclaration('foo', max=8)
        exception = ParameterValueIllegalException(decl, 8.1)
        self.assertEqual("The value 8.1 provided for parameter foo is illegal (min = -inf, max = 8)", str(exception))

if __name__ == "__main__":
    unittest.main(verbosity=2)
