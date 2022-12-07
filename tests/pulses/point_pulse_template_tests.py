import unittest

import numpy as np

from qupulse.pulses.parameters import ParameterNotProvidedException

from qupulse.pulses.point_pulse_template import PointPulseTemplate, PointWaveform, InvalidPointDimension, PointPulseEntry, PointWaveformEntry
from tests.pulses.measurement_tests import ParameterConstrainerTest, MeasurementDefinerTest
from qupulse.pulses.multi_channel_pulse_template import MultiChannelWaveform
from qupulse.pulses.interpolation import HoldInterpolationStrategy, LinearInterpolationStrategy
from qupulse.expressions import Expression, ExpressionScalar
from qupulse.serialization import Serializer
from qupulse.pulses.parameters import ParameterConstraint

from tests.serialization_dummies import DummySerializer, DummyStorageBackend
from tests.serialization_tests import SerializableTests


class PointPulseEntryTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_instantiate(self):
        ppe = PointPulseEntry('t', 'V', HoldInterpolationStrategy())

        l = ppe.instantiate({'t': 1., 'V': np.arange(3.)}, 3)
        expected = (PointWaveformEntry(1., 0, HoldInterpolationStrategy()),
                             PointWaveformEntry(1., 1, HoldInterpolationStrategy()),
                             PointWaveformEntry(1., 2, HoldInterpolationStrategy()))
        self.assertEqual(l, expected)

    def test_invalid_point_exception(self):
        ppe = PointPulseEntry('t', 'V', HoldInterpolationStrategy())

        with self.assertRaises(InvalidPointDimension) as cm:
            ppe.instantiate({'t': 1., 'V': np.ones(3)}, 4)

        self.assertEqual(cm.exception.expected, 4)
        self.assertEqual(cm.exception.received, 3)

    def test_scalar_expansion(self):
        ppe = PointPulseEntry('t', 'V', HoldInterpolationStrategy())

        l = ppe.instantiate({'t': 1., 'V': 0.}, 3)

        self.assertEqual(l, (PointWaveformEntry(1., 0., HoldInterpolationStrategy()),
                             PointWaveformEntry(1., 0., HoldInterpolationStrategy()),
                             PointWaveformEntry(1., 0., HoldInterpolationStrategy())))


class PointPulseTemplateTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_defined_channels(self):
        self.assertEqual(PointPulseTemplate([(1, 'A')], [0]).defined_channels, {0})

        self.assertEqual(PointPulseTemplate([(1, 'A')], [0, 'asd']).defined_channels, {0, 'asd'})

    def test_duration(self):
        self.assertEqual(PointPulseTemplate([(1, 'A')], [0]).duration, 1)

        self.assertEqual(PointPulseTemplate([(1, 'A'), ('t+6', 'B')], [0, 'asd']).duration, 't+6')

    def test_point_parameters(self):
        self.assertEqual(PointPulseTemplate([(1, 'A'), ('t+6', 'B+C')], [0, 'asd']).point_parameters,
                         {'A', 'B', 't', 'C'})

    def test_parameter_names(self):
        self.assertEqual({'a', 'b', 'n', 'A', 'B', 't', 'C'},
                         PointPulseTemplate([(1, 'A'), ('t+6', 'B+C')],
                                            [0, 'asd'],
                                            measurements=[('M', 'n', 1)],
                                            parameter_constraints=['a < b']).parameter_names)

    def test_integral(self) -> None:
        pulse = PointPulseTemplate(
            [(1, (2, 'b'), 'hold'),
             (3, (0, 0), 'linear'),
             (4, (2, 'c'), 'jump'),
             (5, (8, 'd'), 'hold')],
            [0, 'other_channel']
        )
        self.assertEqual({0: ExpressionScalar('2 + 6'),
                          'other_channel': ExpressionScalar('b + b + 2*c')},
                         pulse.integral)

        pulse = PointPulseTemplate(
            [(1, ('2', 'b'), 'hold'), ('t0', (0, 0), 'linear'), (4, (2.0, 'c'), 'jump'), ('g', (8, 'd'), 'hold')],
            ['symbolic', 1]
        )
        self.assertEqual({'symbolic': ExpressionScalar('2 + 2.0*g - 1.0*t0 - 1.0'),
                          1: ExpressionScalar('b + b*(t0 - 1) / 2 + c*(g - 4) + c*(-t0 + 4)')},
                         pulse.integral)

        ppt = PointPulseTemplate([(0, 0), ('t_init', 0)], ['X', 'Y'])
        self.assertEqual(ppt.integral, {'X': 0, 'Y': 0})

        ppt = PointPulseTemplate([(0., 'a'), ('t_1', 'b', 'linear'), ('t_2', (0, 0))], ('X', 'Y'))
        parameters = {'a': (3.4, 4.1), 'b': 4, 't_1': 2, 't_2': 5}
        integral = {ch: v.evaluate_in_scope(parameters) for ch, v in ppt.integral.items()}
        self.assertEqual({'X': 2 * (3.4 + 4) / 2 + (5 - 2) * 4,
                          'Y': 2 * (4.1 + 4) / 2 + (5 - 2) * 4},
                         integral)

    def test_initial_final_values(self):
        pulse = PointPulseTemplate(
            [(1, (2, 'b'), 'hold'),
             (3, (0, 0), 'linear'),
             (4, (2, 'c'), 'jump'),
             (5, (8, 'd'), 'hold')],
            [0, 'other_channel']
        )
        self.assertEqual({0: 2, 'other_channel': 'b'}, pulse.initial_values)
        self.assertEqual({0: 8, 'other_channel': 'd'}, pulse.final_values)

        pulse = PointPulseTemplate(
            [(1, 'b', 'hold'),
             (3, (0, 0), 'linear'),
             (4, (2, 'c'), 'jump'),
             (5, 'd', 'hold')],
            [0, 'other_channel']
        )
        self.assertEqual({0: 'IndexedBroadcast(b, (2,), 0)', 'other_channel': 'IndexedBroadcast(b, (2,), 1)'},
                         pulse.initial_values)
        self.assertEqual({0: 'IndexedBroadcast(d, (2,), 0)', 'other_channel': 'IndexedBroadcast(d, (2,), 1)'},
                         pulse.final_values)


class PointPulseTemplateSequencingTests(unittest.TestCase):
    def test_build_waveform_empty(self):
        self.assertIsNone(PointPulseTemplate([('t1', 'A')], [0]).build_waveform(parameters={'t1': 0, 'A': 1},
                                                                                channel_mapping={0: 1}))

    def test_build_waveform_single_channel(self):
        ppt = PointPulseTemplate([('t1', 'A'),
                                  ('t2', 0., 'hold'),
                                  ('t1+t2', 'B+C', 'linear')], [0])

        parameters = {'t1': 0.1, 't2': 1., 'A': 1., 'B': 2., 'C': 19.}

        wf = ppt.build_waveform(parameters=parameters, channel_mapping={0: 1})
        expected = PointWaveform.from_table(1, [(0, 1., HoldInterpolationStrategy()),
                                     (0.1, 1., HoldInterpolationStrategy()),
                                     (1., 0., HoldInterpolationStrategy()),
                                     (1.1, 21., LinearInterpolationStrategy())])
        self.assertIsInstance(wf, PointWaveform)
        self.assertEqual(wf, expected)

    def test_build_waveform_single_channel_with_measurements(self):
        ppt = PointPulseTemplate([('t1', 'A'),
                                  ('t2', 0., 'hold'),
                                  ('t1+t2', 'B+C', 'linear')], [0], measurements=[('M', 'n', 1), ('L', 'n', 1)])

        parameters = {'t1': 0.1, 't2': 1., 'A': 1., 'B': 2., 'C': 19., 'n': 0.2}
        wf = ppt.build_waveform(parameters=parameters, channel_mapping={0: 1})
        expected = PointWaveform.from_table(1, [(0, 1., HoldInterpolationStrategy()),
                                     (0.1, 1., HoldInterpolationStrategy()),
                                     (1., 0., HoldInterpolationStrategy()),
                                     (1.1, 21., LinearInterpolationStrategy())])
        self.assertEqual(wf, expected)

    def test_build_waveform_multi_channel_same(self):
        ppt = PointPulseTemplate([('t1', 'A'),
                                  ('t2', 0., 'hold'),
                                  ('t1+t2', 'B+C', 'linear')], [0, 'A'], measurements=[('M', 'n', 1), ('L', 'n', 1)])

        parameters = {'t1': 0.1, 't2': 1., 'A': 1., 'B': 2., 'C': 19., 'n': 0.2}
        wf = ppt.build_waveform(parameters=parameters, channel_mapping={0: 1, 'A': 'A'})
        expected_1 = PointWaveform.from_table(1, ((0, 1., HoldInterpolationStrategy()),
                                     (0.1, 1., HoldInterpolationStrategy()),
                                     (1., 0., HoldInterpolationStrategy()),
                                     (1.1, 21., LinearInterpolationStrategy())))
        expected_A = PointWaveform.from_table('A', [(0, 1., HoldInterpolationStrategy()),
                                     (0.1, 1., HoldInterpolationStrategy()),
                                     (1., 0., HoldInterpolationStrategy()),
                                     (1.1, 21., LinearInterpolationStrategy())])
        self.assertEqual(wf.defined_channels, {1, 'A'})
        self.assertEqual(wf._sub_waveforms[0], expected_1)
        self.assertEqual(wf._sub_waveforms[1], expected_A)

    def test_build_waveform_multi_channel_vectorized(self):
        ppt = PointPulseTemplate([('t1', 'A'),
                                  ('t2', 0., 'hold'),
                                  ('t1+t2', 'B+C', 'linear')], [0, 'A'], measurements=[('M', 'n', 1), ('L', 'n', 1)])

        parameters = {'t1': 0.1, 't2': 1., 'A': np.ones(2), 'B': np.arange(2), 'C': 19., 'n': 0.2}
        wf = ppt.build_waveform(parameters=parameters, channel_mapping={0: 1, 'A': 'A'})
        expected_1 = PointWaveform.from_table(1, [(0, 1., HoldInterpolationStrategy()),
                                       (0.1, 1., HoldInterpolationStrategy()),
                                       (1., 0., HoldInterpolationStrategy()),
                                       (1.1, 19., LinearInterpolationStrategy())])
        expected_A = PointWaveform.from_table('A', [(0, 1., HoldInterpolationStrategy()),
                                     (0.1, 1., HoldInterpolationStrategy()),
                                     (1., 0., HoldInterpolationStrategy()),
                                     (1.1, 20., LinearInterpolationStrategy())])
        self.assertEqual(wf.defined_channels, {1, 'A'})
        self.assertEqual(wf._sub_waveforms[0], expected_1)
        self.assertEqual(wf._sub_waveforms[1], expected_A)

    def test_build_waveform_none_channel(self):
        ppt = PointPulseTemplate([('t1', 'A'),
                                  ('t2', 0., 'hold'),
                                  ('t1+t2', 'B+C', 'linear')], [0, 'A', 'C'], measurements=[('M', 'n', 1), ('L', 'n', 1)])
        parameters = {'t1': 0.1, 't2': 1., 'A': np.ones(3), 'B': np.arange(3), 'C': 19., 'n': 0.2}

        self.assertIsNone(ppt.build_waveform(parameters, {0: None, 'A': None, 'C': None}))

        wf = ppt.build_waveform(parameters, {0: 1, 'A': None, 'C': None})
        self.assertIsInstance(wf, PointWaveform)
        self.assertEqual(wf.defined_channels, {1})

        wf = ppt.build_waveform(parameters, {0: 1, 'A': 2, 'C': None})
        self.assertIsInstance(wf, MultiChannelWaveform)
        self.assertEqual(wf.defined_channels, {1, 2})


class TablePulseTemplateConstraintTest(ParameterConstrainerTest):
    def __init__(self, *args, **kwargs):

        def ppt_constructor(parameter_constraints=None):
            return PointPulseTemplate([('t', 'V', 'hold')], channel_names=[0],
                                      parameter_constraints=parameter_constraints, measurements=[('M', 'n', 1)])

        super().__init__(*args,
                         to_test_constructor=ppt_constructor, **kwargs)


class TablePulseTemplateMeasurementTest(MeasurementDefinerTest):
    def __init__(self, *args, **kwargs):

        def tpt_constructor(measurements=None):
            return PointPulseTemplate([('t', 'V', 'hold')], channel_names=[0],
                                      parameter_constraints=['a < b'], measurements=measurements)

        super().__init__(*args,
                         to_test_constructor=tpt_constructor, **kwargs)


class PointPulseTemplateSerializationTests(SerializableTests, unittest.TestCase):

    @property
    def class_to_test(self):
        return PointPulseTemplate

    def make_kwargs(self):
        return {
            'time_point_tuple_list': [('foo', 2, 'hold'), ('hugo', 'A + B', 'linear'), ('sudo', [1, 'a'], 'jump')],
            'channel_names': (0, 'A'),
            'measurements': [('m', 1, 1), ('foo', 'z', 'o')],
            'parameter_constraints': [str(ParameterConstraint('ilse>2')), str(ParameterConstraint('k>foo'))]
        }

    def assert_equal_instance_except_id(self, lhs: PointPulseTemplate, rhs: PointPulseTemplate):
        self.assertIsInstance(lhs, PointPulseTemplate)
        self.assertIsInstance(rhs, PointPulseTemplate)
        self.assertEqual(lhs.point_pulse_entries, rhs.point_pulse_entries)
        self.assertEqual(lhs.measurement_declarations, rhs.measurement_declarations)
        self.assertEqual(lhs.parameter_constraints, rhs.parameter_constraints)
        self.assertEqual(lhs.defined_channels, rhs.defined_channels)


class PointPulseTemplateOldSerializationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.entries = [('foo', 2, 'hold'), ('hugo', 'A + B', 'linear'), ('sudo', [1, 'a'], 'jump')]
        self.measurements = [('m', 1, 1), ('foo', 'z', 'o')]
        self.template = PointPulseTemplate(time_point_tuple_list=self.entries, channel_names=[0, 'A'],
                                           measurements=self.measurements,
                                           identifier='foo', parameter_constraints=['ilse>2', 'k>foo'],
                                           registry=dict())
        self.maxDiff = None

    def test_get_serialization_data_old(self) -> None:
        # test for deprecated version during transition period, remove after final switch
        with self.assertWarnsRegex(DeprecationWarning, "deprecated",
                                   msg="PointPT does not issue warning for old serialization routines."):
            expected_data = dict(measurements=self.measurements,
                                 time_point_tuple_list=self.entries,
                                 channel_names=(0, 'A'),
                                 parameter_constraints=[str(Expression('ilse>2')), str(Expression('k>foo'))])

            serializer = DummySerializer(lambda x: dict(name=x.name), lambda x: x.name, lambda x: x['name'])
            data = self.template.get_serialization_data(serializer)
            self.assertEqual(expected_data, data)

    def test_deserialize_old(self) -> None:
        # test for deprecated version during transition period, remove after final switch
        with self.assertWarnsRegex(DeprecationWarning, "deprecated",
                                   msg="PointPT does not issue warning for old serialization routines."):
            data = dict(measurements=self.measurements,
                        time_point_tuple_list=self.entries,
                        channel_names=(0, 'A'),
                        parameter_constraints=['ilse>2', 'k>foo'],
                        identifier='foo')

            # deserialize
            serializer = DummySerializer(lambda x: dict(name=x.name), lambda x: x.name, lambda x: x['name'])
            template = PointPulseTemplate.deserialize(serializer, **data)

            self.assertEqual(template.point_pulse_entries, self.template.point_pulse_entries)
            self.assertEqual(template.measurement_declarations, self.template.measurement_declarations)
            self.assertEqual(template.parameter_constraints, self.template.parameter_constraints)

    def test_serializer_integration_old(self):
        registry = dict()

        # test for deprecated version during transition period, remove after final switch
        with self.assertWarnsRegex(DeprecationWarning, "deprecated",
                                   msg="PointPT does not issue warning for old serialization routines."):
            serializer = Serializer(DummyStorageBackend())
            serializer.serialize(self.template)
            template = serializer.deserialize('foo')

            self.assertIsInstance(template, PointPulseTemplate)
            self.assertEqual(template.point_pulse_entries, self.template.point_pulse_entries)
            self.assertEqual(template.measurement_declarations, self.template.measurement_declarations)
            self.assertEqual(template.parameter_constraints, self.template.parameter_constraints)


class PointPulseExpressionIntegralTests(unittest.TestCase):
    def setUp(self):
        self.template = PointPulseTemplate(**PointPulseTemplateSerializationTests().make_kwargs())
        self.parameter_sets = [
            {'foo': 1., 'hugo': 2., 'sudo': 3., 'A': 4., 'B': 5., 'a': 6., 'ilse': 7., 'k': 8.},
            {'foo': 1.1, 'hugo': 2.6, 'sudo': 2.7, 'A': np.array([3., 4.]), 'B': 5., 'a': 6., 'ilse': 7., 'k': 8.},
        ]

    def test_integral_as_expression_compatible(self):
        import sympy

        t = self.template._AS_EXPRESSION_TIME
        as_expression = self.template._as_expression()
        integral = self.template.integral
        duration = self.template.duration.underlying_expression

        self.assertEqual(self.template.defined_channels, integral.keys())
        self.assertEqual(self.template.defined_channels, as_expression.keys())

        for channel in self.template.defined_channels:
            ch_expr = as_expression[channel].underlying_expression
            ch_int = integral[channel].underlying_expression

            symbolic = sympy.integrate(ch_expr, (t, 0, duration))
            symbolic = sympy.simplify(symbolic)

            scalar_from_as_expr = ExpressionScalar(symbolic)
            scalar_from_integral = ExpressionScalar(ch_int)

            for parameters in self.parameter_sets:
                num_from_expr = scalar_from_as_expr.evaluate_in_scope(parameters)
                num_from_in = scalar_from_integral.evaluate_in_scope(parameters)
                np.testing.assert_almost_equal(num_from_in, num_from_expr)

            # TODO: the following fails even with a lot of assumptions in sympy 1.6
            # self.assertEqual(ch_int, symbolic)

    def test_as_expression_wf_and_sample_compatible(self):
        as_expression = self.template._as_expression()

        for parameters in self.parameter_sets:
            wf = self.template.build_waveform(parameters, {c: c for c in self.template.defined_channels})

            ts = np.linspace(0, float(wf.duration), num=33)
            sampled = {ch: wf.get_sampled(ch, ts) for ch in self.template.defined_channels}

            from_expr = {}
            for ch, expected_vs in sampled.items():
                ch_expr = as_expression[ch]

                ch_from_expr = []
                for t, expected in zip(ts, expected_vs):
                    result_expr = ch_expr.evaluate_symbolic({**parameters, self.template._AS_EXPRESSION_TIME: t})
                    ch_from_expr.append(result_expr.sympified_expression)
                from_expr[ch] = ch_from_expr

                np.testing.assert_almost_equal(expected_vs, ch_from_expr)


