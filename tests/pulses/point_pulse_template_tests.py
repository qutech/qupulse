import unittest

import numpy as np

from qctoolkit.pulses.parameters import ParameterNotProvidedException

from qctoolkit.pulses.point_pulse_template import PointPulseTemplate, PointWaveform, InvalidPointDimension, PointPulseEntry, PointWaveformEntry
from tests.pulses.measurement_tests import ParameterConstrainerTest, MeasurementDefinerTest
from tests.pulses.sequencing_dummies import DummyParameter, DummyCondition
from qctoolkit.pulses.multi_channel_pulse_template import MultiChannelWaveform
from qctoolkit.pulses.interpolation import HoldInterpolationStrategy, JumpInterpolationStrategy, LinearInterpolationStrategy
from tests.serialization_dummies import DummySerializer, DummyStorageBackend
from qctoolkit.expressions import Expression
from qctoolkit.serialization import Serializer

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
        self.assertEqual(PointPulseTemplate([(1, 'A'), ('t+6', 'B+C')],
                                            [0, 'asd'],
                                            measurements=[('M', 'n', 1)],
                                            parameter_constraints=['a < b']).parameter_names,
                         {'a', 'b', 'n', 'A', 'B', 't', 'C'})

    def test_requires_stop_missing_param(self) -> None:
        table = PointPulseTemplate([('foo', 'v')], [0])
        with self.assertRaises(ParameterNotProvidedException):
            table.requires_stop({'foo': DummyParameter(0, False)}, {})

    def test_requires_stop(self) -> None:
        point = PointPulseTemplate([('foo', 'v'), ('bar', 0)], [0])
        test_sets = [(False, {'foo': DummyParameter(0, False), 'bar': DummyParameter(0, False), 'v': DummyParameter(0, False)}, {'foo': DummyCondition(False)}),
                     (False, {'foo': DummyParameter(0, False), 'bar': DummyParameter(0, False), 'v': DummyParameter(0, False)}, {'foo': DummyCondition(True)}),
                     (True, {'foo': DummyParameter(0, True), 'bar': DummyParameter(0, False), 'v': DummyParameter(0, False)}, {'foo': DummyCondition(False)}),
                     (True, {'foo': DummyParameter(0, True), 'bar': DummyParameter(0, False), 'v': DummyParameter(0, False)}, {'foo': DummyCondition(True)}),
                     (True, {'foo': DummyParameter(0, False), 'bar': DummyParameter(0, False), 'v': DummyParameter(0, True)}, {'foo': DummyCondition(False)}),
                     (True, {'foo': DummyParameter(0, False), 'bar': DummyParameter(0, False), 'v': DummyParameter(0, True)}, {'foo': DummyCondition(True)}),
                     (True, {'foo': DummyParameter(0, True), 'bar': DummyParameter(0, True), 'v': DummyParameter(0, True)}, {'foo': DummyCondition(False)}),
                     (True, {'foo': DummyParameter(0, True), 'bar': DummyParameter(0, True), 'v': DummyParameter(0, True)}, {'foo': DummyCondition(True)})]
        for expected_result, parameter_set, condition_set in test_sets:
            self.assertEqual(expected_result, point.requires_stop(parameter_set, condition_set))

    def test_build_waveform_empty(self):
        self.assertIsNone(PointPulseTemplate([('t1', 'A')], [0]).build_waveform(parameters={'t1': 0, 'A': 1}, channel_mapping={0: 1}, measurement_mapping=dict()))

    def test_build_waveform_single_channel(self):
        ppt = PointPulseTemplate([('t1', 'A'),
                                  ('t2', 0., 'hold'),
                                  ('t1+t2', 'B+C', 'linear')], [0])

        parameters = {'t1': 0.1, 't2': 1., 'A': 1., 'B': 2., 'C': 19.}

        wf = ppt.build_waveform(parameters=parameters, channel_mapping={0: 1}, measurement_mapping=dict())
        expected = PointWaveform(1, [(0, 1., HoldInterpolationStrategy()),
                                     (0.1, 1., HoldInterpolationStrategy()),
                                     (1., 0., HoldInterpolationStrategy()),
                                     (1.1, 21., LinearInterpolationStrategy())], [])
        self.assertIsInstance(wf, PointWaveform)
        self.assertEqual(wf, expected)

    def test_build_waveform_single_channel_with_measurements(self):
        ppt = PointPulseTemplate([('t1', 'A'),
                                  ('t2', 0., 'hold'),
                                  ('t1+t2', 'B+C', 'linear')], [0], measurements=[('M', 'n', 1), ('L', 'n', 1)])

        parameters = {'t1': 0.1, 't2': 1., 'A': 1., 'B': 2., 'C': 19., 'n': 0.2}
        wf = ppt.build_waveform(parameters=parameters, channel_mapping={0: 1}, measurement_mapping={'M': 'K', 'L': 'L'})
        expected = PointWaveform(1, [(0, 1., HoldInterpolationStrategy()),
                                     (0.1, 1., HoldInterpolationStrategy()),
                                     (1., 0., HoldInterpolationStrategy()),
                                     (1.1, 21., LinearInterpolationStrategy())],
                                 [('K', 0.2, 1), ('L', 0.2, 1)])
        self.assertEqual(wf, expected)

    def test_build_waveform_multi_channel_same(self):
        ppt = PointPulseTemplate([('t1', 'A'),
                                  ('t2', 0., 'hold'),
                                  ('t1+t2', 'B+C', 'linear')], [0, 'A'], measurements=[('M', 'n', 1), ('L', 'n', 1)])

        parameters = {'t1': 0.1, 't2': 1., 'A': 1., 'B': 2., 'C': 19., 'n': 0.2}
        wf = ppt.build_waveform(parameters=parameters, channel_mapping={0: 1, 'A': 'A'}, measurement_mapping={'M': 'K', 'L': 'L'})
        expected_1 = PointWaveform(1, [(0, 1., HoldInterpolationStrategy()),
                                     (0.1, 1., HoldInterpolationStrategy()),
                                     (1., 0., HoldInterpolationStrategy()),
                                     (1.1, 21., LinearInterpolationStrategy())],
                                 [('K', 0.2, 1), ('L', 0.2, 1)])
        expected_A = PointWaveform('A', [(0, 1., HoldInterpolationStrategy()),
                                     (0.1, 1., HoldInterpolationStrategy()),
                                     (1., 0., HoldInterpolationStrategy()),
                                     (1.1, 21., LinearInterpolationStrategy())], [])
        self.assertEqual(wf.defined_channels, {1, 'A'})
        self.assertEqual(wf._sub_waveforms[0].defined_channels, {1})
        self.assertEqual(wf._sub_waveforms[0], expected_1)
        self.assertEqual(wf._sub_waveforms[1].defined_channels, {'A'})
        self.assertEqual(wf._sub_waveforms[1], expected_A)

    def test_build_waveform_multi_channel_vectorized(self):
        ppt = PointPulseTemplate([('t1', 'A'),
                                  ('t2', 0., 'hold'),
                                  ('t1+t2', 'B+C', 'linear')], [0, 'A'], measurements=[('M', 'n', 1), ('L', 'n', 1)])

        parameters = {'t1': 0.1, 't2': 1., 'A': np.ones(2), 'B': np.arange(2), 'C': 19., 'n': 0.2}
        wf = ppt.build_waveform(parameters=parameters, channel_mapping={0: 1, 'A': 'A'}, measurement_mapping={'M': 'K', 'L': 'L'})
        expected_1 = PointWaveform(1, [(0, 1., HoldInterpolationStrategy()),
                                       (0.1, 1., HoldInterpolationStrategy()),
                                       (1., 0., HoldInterpolationStrategy()),
                                       (1.1, 19., LinearInterpolationStrategy())],
                                 [('K', 0.2, 1), ('L', 0.2, 1)])
        expected_A = PointWaveform('A', [(0, 1., HoldInterpolationStrategy()),
                                     (0.1, 1., HoldInterpolationStrategy()),
                                     (1., 0., HoldInterpolationStrategy()),
                                     (1.1, 20., LinearInterpolationStrategy())], [])
        self.assertEqual(wf.defined_channels, {1, 'A'})
        self.assertEqual(wf._sub_waveforms[0].defined_channels, {1})
        self.assertEqual(wf._sub_waveforms[0], expected_1)
        self.assertEqual(wf._sub_waveforms[1].defined_channels, {'A'})
        self.assertEqual(wf._sub_waveforms[1], expected_A)


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


class PointPulseTemplateSerializationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.serializer = DummySerializer(lambda x: dict(name=x.name), lambda x: x.name, lambda x: x['name'])
        self.entries = [('foo', 2, 'hold'), ('hugo', 'A + B', 'linear')]
        self.measurements = [('m', 1, 1), ('foo', 'z', 'o')]
        self.template = PointPulseTemplate(time_point_tuple_list=self.entries, channel_names=[0, 'A'],
                                           measurements=self.measurements,
                                           identifier='foo', parameter_constraints=['ilse>2', 'k>foo'])
        self.expected_data = dict(type=self.serializer.get_type_identifier(self.template))
        self.maxDiff = None

    def test_get_serialization_data(self) -> None:
        expected_data = dict(measurements=self.measurements,
                             time_point_tuple_list=self.entries,
                             channel_names=(0, 'A'),
                             parameter_constraints=[str(Expression('ilse>2')), str(Expression('k>foo'))])

        data = self.template.get_serialization_data(self.serializer)
        self.assertEqual(expected_data, data)

    def test_deserialize(self) -> None:
        data = dict(measurements=self.measurements,
                    time_point_tuple_list=self.entries,
                    channel_names=(0, 'A'),
                    parameter_constraints=['ilse>2', 'k>foo'],
                    identifier='foo')

        # deserialize
        template = PointPulseTemplate.deserialize(self.serializer, **data)

        self.assertEqual(template.point_pulse_entries, self.template.point_pulse_entries)
        self.assertEqual(template.measurement_declarations, self.template.measurement_declarations)
        self.assertEqual(template.parameter_constraints, self.template.parameter_constraints)

    def test_serializer_integration(self):
        serializer = Serializer(DummyStorageBackend())
        serializer.serialize(self.template)
        template = serializer.deserialize('foo')

        self.assertIsInstance(template, PointPulseTemplate)
        self.assertEqual(template.point_pulse_entries, self.template.point_pulse_entries)
        self.assertEqual(template.measurement_declarations, self.template.measurement_declarations)
        self.assertEqual(template.parameter_constraints, self.template.parameter_constraints)