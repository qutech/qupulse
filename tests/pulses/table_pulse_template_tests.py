import unittest
import warnings

import numpy
import sympy

from qupulse.expressions import Expression, ExpressionScalar
from qupulse.serialization import Serializer
from qupulse.pulses.table_pulse_template import TablePulseTemplate, TableWaveform, TableEntry, TableWaveformEntry, ZeroDurationTablePulseTemplate, AmbiguousTablePulseEntry, concatenate
from qupulse.pulses.parameters import ParameterNotProvidedException, ParameterConstraintViolation, ParameterConstraint
from qupulse.pulses.interpolation import HoldInterpolationStrategy, LinearInterpolationStrategy, JumpInterpolationStrategy
from qupulse.pulses.multi_channel_pulse_template import MultiChannelWaveform

from tests.pulses.sequencing_dummies import DummyInterpolationStrategy, DummyPulseTemplate
from tests.serialization_dummies import DummySerializer, DummyStorageBackend
from tests.pulses.measurement_tests import ParameterConstrainerTest, MeasurementDefinerTest
from tests.serialization_tests import SerializableTests


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

    def test_slots(self):
        entry = TableEntry('a', Expression.make('b'), 'hold')

        self.assertFalse(hasattr(entry, '__dict__'))

    def test_unknown_interpolation_strategy(self):
        with self.assertRaises(KeyError):
            TableEntry(0, 0, 'foo')

    def test_sequence_integral(self):
        def get_sympy(v):
            return v.sympified_expression

        entries = [TableEntry(0, 0), TableEntry(1, 0, 'hold')]
        self.assertEqual(ExpressionScalar(0), TableEntry._sequence_integral(entries, get_sympy))

        entries = [TableEntry(0, 1), TableEntry(1, 1, 'hold')]
        self.assertEqual(ExpressionScalar(1), TableEntry._sequence_integral(entries, get_sympy))

        entries = [TableEntry(0, 0), TableEntry(1, 1, 'linear')]
        self.assertEqual(ExpressionScalar(.5), TableEntry._sequence_integral(entries, get_sympy))

        entries = [TableEntry('t0', 'a', 'linear'), TableEntry('t1', 'b', 'linear'), TableEntry('t2', 'c', 'hold')]
        self.assertEqual(ExpressionScalar('(t1-t0)*(a+b)/2 + (t2-t1)*b'),
                         TableEntry._sequence_integral(entries, get_sympy))

    def test_sequence_as_expression(self):
        def get_sympy(v):
            return v.sympified_expression

        t = sympy.Dummy('t')

        times = {
            t: 0.5,
            't0': 0.3,
            't1': 0.7,
            't2': 1.3,
        }

        entries = [TableEntry(0, 0, None), TableEntry(1, 0, 'hold')]
        self.assertEqual(ExpressionScalar(0),
                         TableEntry._sequence_as_expression(entries, get_sympy, t, pre_value=None, post_value=None).sympified_expression.subs(times))

        entries = [TableEntry(0, 1, None), TableEntry(1, 1, 'hold')]
        self.assertEqual(ExpressionScalar(1),
                         TableEntry._sequence_as_expression(entries, get_sympy, t, pre_value=None, post_value=None).sympified_expression.subs(times))

        entries = [TableEntry(0, 0, None), TableEntry(1, 1, 'linear')]
        self.assertEqual(ExpressionScalar(.5),
                         TableEntry._sequence_as_expression(entries, get_sympy, t, pre_value=None, post_value=None).sympified_expression.subs(times))

        entries = [TableEntry('t0', 'a', 'linear'),
                   TableEntry('t1', 'b', 'linear'),
                   TableEntry('t2', 'c', 'hold')]
        self.assertEqual(ExpressionScalar('(a+b)*.5'),
                         TableEntry._sequence_as_expression(entries, get_sympy, t, pre_value=None, post_value=None).sympified_expression.subs(times))


class TablePulseTemplateTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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

        with self.assertRaises(ValueError):
            TablePulseTemplate({0: [('a', 1),
                                    (2, 0)]}, parameter_constraints=['2>3'])

    @unittest.skip(reason='Needs a better inequality solver')
    def test_time_not_increasing_hard(self):
        with self.assertRaises(ValueError):
            TablePulseTemplate({0: [('a*c', 3),
                                    ('b', 1),
                                    ('c*a', 'k')]}, parameter_constraints=['a*c < b'])

    def test_parameter_names(self) -> None:
        table = TablePulseTemplate({'a': [('foo', 'bar')]}, parameter_constraints=['foo < hugo'], measurements=[('meas', 'd', 2)])
        self.assertEqual({'foo', 'bar', 'hugo', 'd'}, table.parameter_names)

    def test_time_is_0_on_construction(self) -> None:
        with self.assertWarns(ZeroDurationTablePulseTemplate):
            warnings.simplefilter('default', ZeroDurationTablePulseTemplate)
            table = TablePulseTemplate({0: [(0, 1.4)]})
            self.assertTrue(table.duration == 0)
        self.assertTrue(table.duration == 0)

        self.assertIsNone(table.build_waveform(parameters=dict(),
                                               channel_mapping={0: 0}))

    def test_time_is_0_on_instantiation(self):
        table = TablePulseTemplate({0: [('a', 1)]})
        self.assertEqual(table.duration, Expression('a'))
        self.assertEqual(table.parameter_names, {'a'})

        self.assertIsNone(table.build_waveform(parameters=dict(a=0),
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
                                 channel_mapping={0: 0, 1: 1})
        with self.assertRaises(ParameterConstraintViolation):
            table.build_waveform(parameters=dict(v=1, w=2, t=0.1, x=1.2, y=2),
                                 channel_mapping={0: 0, 1: 1})
        with self.assertRaises(ParameterConstraintViolation):
            table.build_waveform(parameters=dict(v=1, w=2, t=3, x=1.2, y=1),
                                 channel_mapping={0: 0, 1: 1})
        table.build_waveform(parameters=dict(v=1, w=2, t=0.1, x=1.2, y=1),
                             channel_mapping={0: 0, 1: 1})

    def test_external_constraints(self):
        table = TablePulseTemplate({0: [(1, 'v'), (2, 'w')],
                                    1: [('t', 'x'), ('t+2', 'y')]},
                                   parameter_constraints=['x<h', 'y<w', 't<1'])
        self.assertEqual(table.parameter_names, {'v', 'w', 't', 'x', 'y', 'h'})

        with self.assertRaises(ParameterConstraintViolation):
            table.build_waveform(parameters=dict(v=1., w=2, t=0.1, x=2.2, y=1, h=1),
                                 channel_mapping={0: 0, 1: 1})
        table.build_waveform(parameters=dict(v=1., w=2, t=0.1, x=1.2, y=1, h=2),
                             channel_mapping={0: 0, 1: 1})

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
        with self.assertRaisesRegex(ValueError, 'empty TablePulseTemplate'):
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

    def test_from_array_exceptions(self):
        with self.assertRaises(ValueError):
            TablePulseTemplate.from_array(numpy.arange(0), numpy.arange(1), [0])

        with self.assertRaises(ValueError):
            TablePulseTemplate.from_array(numpy.arange(1), numpy.arange(0), [0])

        with self.assertRaises(ValueError):
            TablePulseTemplate.from_array(numpy.array(numpy.ndindex((1, 2, 1))), numpy.arange(2), [0])

        with self.assertRaises(ValueError):
            TablePulseTemplate.from_array(numpy.zeros(3),
                                          numpy.zeros((3, 2, 3)),
                                          [3, 4, 5])

        with self.assertRaises(ValueError):
            TablePulseTemplate.from_array(numpy.zeros((4, 2)),
                                          numpy.zeros((3, 4)), [3, 4, 5])

        with self.assertRaises(ValueError):
            TablePulseTemplate.from_array(numpy.zeros((3, 2)),
                                          numpy.array(numpy.ndindex((4, 6))), [3, 4, 5])

        with self.assertRaises(ValueError):
            TablePulseTemplate.from_array(numpy.zeros((3, 5)),
                                          numpy.array(numpy.ndindex((3, 6))), [3, 4, 5])

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

    def test_from_entry_list_exceptions(self):
        TablePulseTemplate.from_entry_list([(1, 2, 3, 'hold'), (2, 2, 2)], channel_names=['A', 'B'])

        with self.assertRaises(ValueError):
            TablePulseTemplate.from_entry_list([(1, 2, 3, 'hold'), (2, 2)])

        with self.assertRaises(ValueError):
            TablePulseTemplate.from_entry_list([(1, 2, 3, 'hold'), (2, 2, 2)], channel_names=['A'])

        with self.assertWarns(AmbiguousTablePulseEntry):
            TablePulseTemplate.from_entry_list([(1, 2, 3, 'hold'), (2, 2, 'linear')], channel_names=['A', 'B'])

    def test_from_entry_list(self):
        entries = {0: [(0, 9, HoldInterpolationStrategy()),
                       (1, 2, HoldInterpolationStrategy()),
                       (4, 1, LinearInterpolationStrategy())],
                   1: [(0, 8, HoldInterpolationStrategy()),
                       (1, 1, HoldInterpolationStrategy()),
                       (4, 2, LinearInterpolationStrategy())],
                   2: [(0, 7, HoldInterpolationStrategy()),
                       (1, 3, HoldInterpolationStrategy()),
                       (4, 3, LinearInterpolationStrategy())]}

        tpt = TablePulseTemplate.from_entry_list([(0, 9, 8, 7),
                                                  (1, 2, 1, 3, 'hold'),
                                                  (4, 1, 2, 3, 'linear')],
                                                 identifier='tpt')
        self.assertEqual(tpt.entries, entries)
        self.assertEqual(tpt.identifier, 'tpt')

        tpt = TablePulseTemplate.from_entry_list([(0, 9, 8, 7, 'hold'),
                                                  (1, 2, 1, 3, 'hold'),
                                                  (4, 1, 2, 3, 'linear')],
                                                 identifier='tpt2')
        self.assertEqual(tpt.entries, entries)

        entries = {k: entries[i]
                   for k, i in zip('ABC', [0, 1, 2])}
        tpt = TablePulseTemplate.from_entry_list([(0, 9, 8, 7),
                                                  (1, 2, 1, 3, 'hold'),
                                                  (4, 1, 2, 3, 'linear')],
                                                 identifier='tpt3',
                                                 channel_names=['A', 'B', 'C'])
        self.assertEqual(tpt.entries, entries)
        self.assertEqual(tpt.identifier, 'tpt3')

        entries = {0: [(0, 9, HoldInterpolationStrategy()),
                       (1, 2, HoldInterpolationStrategy()),
                       (4, 1, HoldInterpolationStrategy())],
                   1: [(0, 8, HoldInterpolationStrategy()),
                       (1, 1, HoldInterpolationStrategy()),
                       (4, 2, HoldInterpolationStrategy())],
                   2: [(0, 7, HoldInterpolationStrategy()),
                       (1, 3, HoldInterpolationStrategy()),
                       (4, 3, HoldInterpolationStrategy())]}
        tpt = TablePulseTemplate.from_entry_list([(0, 9, 8, 7),
                                                  (1, 2, 1, 3),
                                                  (4, 1, 2, 3)],
                                                 identifier='tpt4')
        self.assertEqual(tpt.entries, entries)

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

    def test_measurement_names(self):
        tpt = TablePulseTemplate({0: [(10, 1)]}, measurements=[('A', 2, 3), ('AB', 0, 1)])
        self.assertEqual(tpt.measurement_names, {'A', 'AB'})

    def test_identifier(self) -> None:
        identifier = 'some name'
        pulse = TablePulseTemplate(entries={0: [(1, 0)]}, identifier=identifier)
        self.assertEqual(pulse.identifier, identifier)

    def test_integral(self) -> None:
        pulse = TablePulseTemplate(entries={0: [(1, 2), (3, 0, 'linear'), (4, 2, 'jump'), (5, 8, 'hold')],
                                            'other_channel': [(0, 7), (2, 0, 'linear'), (10, 0)],
                                            'symbolic': [(3, 'a'), ('b', 4, 'hold'), ('c', Expression('d'), 'linear')]})
        expected = {0: Expression('2 + 2 + 2 + 2 + (Max(c, 10) - 5) * 8'),
                    'other_channel': Expression(7),
                    'symbolic': Expression('3 * a + (b-3)*a + (c-b)*(d+4) / 2 + (Max(10, c) - c) * d')}

        self.assertEqual(expected, pulse.integral)

    def test_initial_final_values(self):
        pulse = TablePulseTemplate(entries={0: [(1, 2), (3, 0, 'linear'), (4, 2, 'jump'), (5, 8, 'hold')],
                                            'other_channel': [(0, 7), (2, 0, 'linear'), (10, 0)],
                                            'symbolic': [(3, 'a'), ('b', 4, 'hold'), ('c', Expression('d'), 'linear')]})
        self.assertEqual({0: 2, 'other_channel': 7, 'symbolic': 'a'}, pulse.initial_values)
        self.assertEqual({0: 8, 'other_channel': 0, 'symbolic': 'd'}, pulse.final_values)

    def test_as_expression(self):
        pulse = TablePulseTemplate(entries={0: [(0, 0), (1, 2), (3, 0, 'linear'), (4, 2, 'jump'), (5, 8, 'hold')],
                                            'other_channel': [(0, 7), (2, 0, 'linear'), (10, 0)],
                                            'symbolic': [(3, 'a'), ('b', 4, 'hold'),
                                                         ('c', Expression('d'), 'linear')]})
        parameters = dict(a=2., b=4, c=9, d=8)
        wf = pulse.build_waveform(parameters, channel_mapping={0: 0,
                                                               'other_channel': 'other_channel',
                                                               'symbolic': 'symbolic'})
        expr = pulse._as_expression()
        ts = numpy.linspace(0, float(wf.duration), num=33)
        sampled = {ch: wf.get_sampled(ch, ts) for ch in pulse.defined_channels}

        from_expr = {}
        for ch, expected_vs in sampled.items():
            ch_expr = expr[ch]

            ch_from_expr = []
            for t, expected in zip(ts, expected_vs):
                params = {**parameters, TablePulseTemplate._AS_EXPRESSION_TIME: t}
                result = ch_expr.sympified_expression.subs(params, simultaneous=True)
                ch_from_expr.append(result)
            from_expr[ch] = ch_from_expr

            numpy.testing.assert_almost_equal(expected_vs, ch_from_expr)



class TablePulseTemplateConstraintTest(ParameterConstrainerTest):
    def __init__(self, *args, **kwargs):

        def tpt_constructor(parameter_constraints=None):
            return TablePulseTemplate({0: [('a', 'b')]},
                                      parameter_constraints=parameter_constraints, measurements=[('M', 'n', 1)])

        super().__init__(*args,
                         to_test_constructor=tpt_constructor, **kwargs)


class TablePulseTemplateMeasurementTest(MeasurementDefinerTest):
    def __init__(self, *args, **kwargs):

        def tpt_constructor(measurements=None):
            return TablePulseTemplate({0: [('a', 'b')]},
                                      parameter_constraints=['a < b'], measurements=measurements)

        super().__init__(*args,
                         to_test_constructor=tpt_constructor, **kwargs)


class TablePulseTemplateSerializationTests(SerializableTests, unittest.TestCase):

    @property
    def class_to_test(self):
        return TablePulseTemplate

    def make_kwargs(self):
        return {
            'entries': dict(A=[('foo', 2, 'hold'), ('hugo', 'ilse', 'linear')],
                            B=[(0, 5, 'hold'), (1, 7, 'jump'), ('k', 't', 'hold')]),
            'measurements': [('m', 1, 1), ('foo', 'z', 'o')],
            'parameter_constraints': [str(ParameterConstraint('ilse>2')), str(ParameterConstraint('k>foo'))]
        }

    def assert_equal_instance_except_id(self, lhs: TablePulseTemplate, rhs: TablePulseTemplate):
        self.assertIsInstance(lhs, TablePulseTemplate)
        self.assertIsInstance(rhs, TablePulseTemplate)
        self.assertEqual(lhs.entries, rhs.entries)
        self.assertEqual(lhs.measurement_declarations, rhs.measurement_declarations)
        self.assertEqual(lhs.parameter_constraints, rhs.parameter_constraints)


class TablePulseTemplateOldSerializationTests(unittest.TestCase):

    def setUp(self) -> None:
        # test for deprecated version during transition period, remove after final switch
        with self.assertWarnsRegex(DeprecationWarning, "deprecated",
                                   msg="TablePT does not issue warning for old serialization routines."):
            self.serializer = DummySerializer(lambda x: dict(name=x.name), lambda x: x.name, lambda x: x['name'])
            self.entries = dict(A=[('foo', 2, 'hold'), ('hugo', 'ilse', 'linear')],
                                B=[(0, 5, 'hold'), (1, 7, 'jump'), ('k', 't', 'hold')])
            self.measurements = [('m', 1, 1), ('foo', 'z', 'o')]
            self.template = TablePulseTemplate(entries=self.entries,
                                               measurements=self.measurements,
                                               identifier='foo', parameter_constraints=['ilse>2', 'k>foo'],
                                               registry=dict())
            self.expected_data = dict(type=self.serializer.get_type_identifier(self.template))
            self.maxDiff = None

    def test_get_serialization_data_old(self) -> None:
        # test for deprecated version during transition period, remove after final switch
        with self.assertWarnsRegex(DeprecationWarning, "deprecated",
                                   msg="TablePT does not issue warning for old serialization routines."):
            expected_data = dict(measurements=self.measurements,
                                 entries=self.entries,
                                 parameter_constraints=[str(Expression('ilse>2')), str(Expression('k>foo'))])

            data = self.template.get_serialization_data(self.serializer)
            self.assertEqual(expected_data, data)

    def test_deserialize_old(self) -> None:
        registry = dict()

        # test for deprecated version during transition period, remove after final switch
        with self.assertWarnsRegex(DeprecationWarning, "deprecated",
                                   msg="TablePT does not issue warning for old serialization routines."):
            data = dict(measurements=self.measurements,
                        entries=self.entries,
                        parameter_constraints=['ilse>2', 'k>foo'],
                        identifier='foo')

            # deserialize
            template = TablePulseTemplate.deserialize(self.serializer, **data, registry=registry)

            self.assertEqual(template.entries, self.template.entries)
            self.assertEqual(template.measurement_declarations, self.template.measurement_declarations)
            self.assertEqual(template.parameter_constraints, self.template.parameter_constraints)

    def test_serializer_integration_old(self):
        registry = dict()

        # test for deprecated version during transition period, remove after final switch
        with self.assertWarnsRegex(DeprecationWarning, "deprecated",
                                   msg="TablePT does not issue warning for old serialization routines."):
            serializer = Serializer(DummyStorageBackend())
            serializer.serialize(self.template)
            template = serializer.deserialize('foo')

            self.assertIsInstance(template, TablePulseTemplate)
            self.assertEqual(template.entries, self.template.entries)
            self.assertEqual(template.measurement_declarations, self.template.measurement_declarations)
            self.assertEqual(template.parameter_constraints, self.template.parameter_constraints)


class TablePulseTemplateSequencingTests(unittest.TestCase):
    def test_build_waveform_single_channel(self):
        table = TablePulseTemplate({0: [(0, 0),
                                        ('foo', 'v', 'linear'),
                                        ('bar', 0, 'jump')]},
                                   parameter_constraints=['foo>1'],
                                   measurements=[('M', 'b', 'l'),
                                                 ('N', 1, 2)])

        parameters = {'v': 2.3, 'foo': 1, 'bar': 4, 'b': 2, 'l': 1}
        channel_mapping = {0: 'ch'}

        with self.assertRaises(ParameterConstraintViolation):
            table.build_waveform(parameters=parameters,
                                 channel_mapping=channel_mapping)

        parameters['foo'] = 1.1
        waveform = table.build_waveform(parameters=parameters,
                                        channel_mapping=channel_mapping)

        self.assertIsInstance(waveform, TableWaveform)
        self.assertEqual(waveform._table,
                         ((0, 0, HoldInterpolationStrategy()),
                          (1.1, 2.3, LinearInterpolationStrategy()),
                          (4, 0, JumpInterpolationStrategy())))
        self.assertEqual(waveform._channel_id,
                         'ch')

    def test_build_waveform_time_type(self):
        from qupulse.utils.types import TimeType

        table = TablePulseTemplate({0: [(0, 0),
                                        ('foo', 'v', 'linear'),
                                        ('bar', 0, 'jump')]},
                                   parameter_constraints=['foo>1'],
                                   measurements=[('M', 'b', 'l'),
                                                 ('N', 1, 2)])

        parameters = {'v': 2.3,
                      'foo': TimeType.from_float(1.), 'bar': TimeType.from_float(4),
                      'b': TimeType.from_float(2), 'l': TimeType.from_float(1)}
        channel_mapping = {0: 'ch'}

        with self.assertRaises(ParameterConstraintViolation):
            table.build_waveform(parameters=parameters,
                                 channel_mapping=channel_mapping)

        parameters['foo'] = TimeType.from_float(1.1)
        waveform = table.build_waveform(parameters=parameters,
                                        channel_mapping=channel_mapping)

        self.assertIsInstance(waveform, TableWaveform)
        self.assertEqual(waveform._table,
                         ((0, 0, HoldInterpolationStrategy()),
                          (TimeType.from_float(1.1), 2.3, LinearInterpolationStrategy()),
                          (4, 0, JumpInterpolationStrategy())))
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
        channel_mapping = {0: 'ch', 3: 'oh'}

        with self.assertRaises(ParameterConstraintViolation):
            table.build_waveform(parameters=parameters,
                                 channel_mapping=channel_mapping)

        parameters['foo'] = 1.1
        waveform = table.build_waveform(parameters=parameters,
                                        channel_mapping=channel_mapping)

        self.assertIsInstance(waveform, MultiChannelWaveform)

        expected_waveforms = [
            TableWaveform.from_table('ch', ((0, 0, HoldInterpolationStrategy()),
                                  (1.1, 2.3, LinearInterpolationStrategy()),
                                  (4, 0, JumpInterpolationStrategy()),
                                  (5.1, 0, HoldInterpolationStrategy()))),
            TableWaveform.from_table('oh', ((0, 1, HoldInterpolationStrategy()),
                                  (5.1, 0, LinearInterpolationStrategy()))),
        ]

        self.assertEqual(waveform._sub_waveforms, tuple(expected_waveforms))

    def test_build_waveform_none(self) -> None:
        table = TablePulseTemplate({0: [(0, 0),
                                        ('foo', 'v', 'linear'),
                                        ('bar', 0, 'jump')],
                                    3: [(0, 1),
                                        ('bar+foo', 0, 'linear')]},
                                   parameter_constraints=['foo>1'],
                                   measurements=[('M', 'b', 'l'),
                                                 ('N', 1, 2)])

        parameters = {'v': 2.3, 'foo': 1, 'bar': 4, 'b': 2, 'l': 1}
        channel_mapping = {0: None, 3: None}

        with self.assertRaises(ParameterConstraintViolation):
            table.build_waveform(parameters=parameters,
                                 channel_mapping=channel_mapping)

        parameters['foo'] = 1.1
        self.assertIsNone(table.build_waveform(parameters=parameters,
                                               channel_mapping=channel_mapping))
        channel_mapping = {0: 1, 3: None}
        wf = table.build_waveform(parameters=parameters,
                                  channel_mapping=channel_mapping)
        self.assertEqual(wf.defined_channels, {1})

    def test_build_waveform_empty(self) -> None:
        table = TablePulseTemplate(dict(a=[('t', 0)]))
        self.assertIsNone(table.build_waveform(dict(t=0), dict(a='a')))


class TablePulseConcatenationTests(unittest.TestCase):
    def test_simple_concatenation(self):
        tpt_1 = TablePulseTemplate({'A': [(0, 1), ('a', 5, 'linear')],
                                    'B': [(0, 2), ('b', 7)]})

        tpt_2 = TablePulseTemplate({'A': [('c', 9), ('a', 10, 'jump')],
                                    'B': [(0, 6),   ('b', 8)]})

        expected = TablePulseTemplate({'A': [(0, 1),
                                             ('a', 5, 'linear'),
                                             ('Max(a, b)', 5),
                                             ('Max(a, b)', 9),
                                             ('Max(a, b) + c', 9),
                                             ('Max(a, b) + a', 10, 'jump')],
                                       'B': [(0,   2),
                                             ('b', 7),
                                             ('Max(a, b)', 7, 'hold'),
                                             ('Max(a, b)', 6),
                                             ('Max(a, b) + b', 8)]})

        concatenated = concatenate(tpt_1, tpt_2)

        self.assertEqual(expected.entries, concatenated.entries)

    def test_triple_concatenation(self):
        tpt_1 = TablePulseTemplate({'A': [(0, 1), ('a', 5, 'linear')],
                                    'B': [(0, 2), ('b', 7)]})

        tpt_2 = TablePulseTemplate({'A': [('c', 9), ('a', 10, 'jump')],
                                    'B': [(0, 6),   ('b', 8)]})

        tpt_3 = TablePulseTemplate({'A': [('fg', 19), ('ab', 110, 'jump')],
                                    'B': [('df', 16), ('ab', 18)]})

        expected = TablePulseTemplate({'A': [(0, 1),
                                             ('a', 5, 'linear'),
                                             ('Max(a, b)', 5),
                                             ('Max(a, b)', 9),
                                             ('Max(a, b) + c', 9),
                                             ('Max(a, b) + a', 10, 'jump'),
                                             ('2*Max(a, b)', 10),
                                             ('2*Max(a, b)', 19),
                                             ('2*Max(a, b) + fg', 19),
                                             ('2*Max(a, b) + ab', 110, 'jump')],
                                       'B': [(0,   2),
                                             ('b', 7),
                                             ('Max(a, b)', 7, 'hold'),
                                             ('Max(a, b)', 6),
                                             ('Max(a, b) + b', 8),
                                             ('2*Max(a, b)', 8),
                                             ('2*Max(a, b)', 16),
                                             ('2*Max(a, b) + df', 16),
                                             ('2*Max(a, b) + ab', 18)]})

        concatenated = concatenate(tpt_1, tpt_2, tpt_3, identifier='asdf')

        self.assertEqual(expected.entries, concatenated.entries)
        self.assertEqual(concatenated.identifier, 'asdf')

    def test_duplication(self):
        tpt = TablePulseTemplate({'A': [(0, 1), ('a', 5)],
                                  'B': [(0, 2), ('b', 3)]})

        concatenated = concatenate(tpt, tpt)

        self.assertIsNot(concatenated.entries, tpt.entries)

        expected = TablePulseTemplate({'A': [(0, 1), ('a', 5), ('Max(a, b)', 5), ('Max(a, b)', 1), ('Max(a, b) + a', 5)],
                                       'B': [(0, 2), ('b', 3), ('Max(a, b)', 3), ('Max(a, b)', 2), ('Max(a, b) + b', 3)]})

        self.assertEqual(expected.entries, concatenated.entries)

    def test_wrong_channels(self):
        tpt_1 = TablePulseTemplate({'A': [(0, 1), ('a', 5, 'linear')],
                                    'B': [(0, 2), ('b', 7)]})

        tpt_2 = TablePulseTemplate({'A': [('c', 9), ('a', 10, 'jump')],
                                    'C': [(0, 6), ('b', 8)]})

        with self.assertRaisesRegex(ValueError, 'differing defined channels'):
            concatenate(tpt_1, tpt_2)

    def test_wrong_type(self):
        dummy = DummyPulseTemplate()
        tpt = TablePulseTemplate({'A': [(0, 1), ('a', 5, 'linear')],
                                  'B': [(0, 2), ('b', 7)]})

        with self.assertRaisesRegex(TypeError, 'not a TablePulseTemplate'):
            concatenate(dummy, tpt)


if __name__ == "__main__":
    unittest.main(verbosity=2)
