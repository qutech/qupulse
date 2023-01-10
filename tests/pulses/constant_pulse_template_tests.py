import unittest

import qupulse.plotting
import qupulse._program.waveforms
import qupulse.utils.sympy
from qupulse.pulses import TablePT, FunctionPT, AtomicMultiChannelPT, MappingPT
from qupulse.pulses.multi_channel_pulse_template import AtomicMultiChannelPulseTemplate
from qupulse.plotting import plot
from qupulse.pulses.sequence_pulse_template import SequencePulseTemplate
from qupulse._program._loop import make_compatible
from qupulse._program.waveforms import ConstantWaveform

from qupulse.serialization import DictBackend, PulseStorage
from qupulse.pulses.constant_pulse_template import ConstantPulseTemplate, ExpressionScalar, TimeType

from tests.serialization_tests import SerializableTests

class TestConstantPulseTemplate(unittest.TestCase):

    def test_ConstantPulseTemplate(self):
        pt = ConstantPulseTemplate(100, {'P1': .5, 'P2': .25})
        self.assertEqual(pt.integral, {'P1': 50, 'P2': 25})

        data = pt.get_serialization_data()
        self.assertEqual(data['name'], pt._name)

        self.assertIn('ConstantPulseTemplate', str(pt))
        self.assertIn('ConstantPulseTemplate', repr(pt))

        self.assertEqual({'P1': .5, 'P2': .25}, pt.initial_values)
        self.assertEqual({'P1': .5, 'P2': .25}, pt.final_values)

    def test_zero_duration(self):
        p1 = ConstantPulseTemplate(10, {'P1': 1.})
        p2 = ConstantPulseTemplate(0, {'P1': 1.})
        p3 = ConstantPulseTemplate(2, {'P1': 1.})

        _ = qupulse.plotting.render(p1.create_program())

        pulse = SequencePulseTemplate(p1, p2, p3)
        prog = pulse.create_program()
        _ = qupulse.plotting.render(prog)

        self.assertEqual(pulse.duration, 12)

    def test_regression_duration_conversion(self):
        old_value = qupulse._program.waveforms.PULSE_TO_WAVEFORM_ERROR

        try:
            qupulse._program.waveforms.PULSE_TO_WAVEFORM_ERROR = 1e-6
            for duration_in_samples in [64, 936320, 24615392]:
                p = ConstantPulseTemplate(duration_in_samples / 2.4, {'a': 0})
                number_of_samples = p.create_program().duration * 2.4
                make_compatible(p.create_program(), 8, 8, 2.4)
                self.assertEqual(number_of_samples.denominator, 1)

                p2 = ConstantPulseTemplate((duration_in_samples + 1) / 2.4, {'a': 0})
                self.assertNotEqual(p.create_program().duration, p2.create_program().duration)
        finally:
            qupulse._program.waveforms.PULSE_TO_WAVEFORM_ERROR = old_value

    def test_regression_duration_conversion_functionpt(self):
        old_value = qupulse._program.waveforms.PULSE_TO_WAVEFORM_ERROR

        try:
            qupulse._program.waveforms.PULSE_TO_WAVEFORM_ERROR = 1e-6
            for duration_in_samples in [64, 2000, 936320]:
                p = FunctionPT('1', duration_expression=duration_in_samples / 2.4, channel='a')
                number_of_samples = p.create_program().duration * 2.4
                self.assertEqual(number_of_samples.denominator, 1)
        finally:
            qupulse._program.waveforms.PULSE_TO_WAVEFORM_ERROR = old_value

    def test_regression_template_combination(self):
        old_value = qupulse.utils.sympy.SYMPY_DURATION_ERROR_MARGIN

        try:
            qupulse.utils.sympy.SYMPY_DURATION_ERROR_MARGIN = 1e-9
            duration_in_seconds = 2e-6
            full_template = ConstantPulseTemplate(duration=duration_in_seconds * 1e9, amplitude_dict={'C1': 1.1})
            duration_in_seconds_derived = 1e-9 * full_template.duration
            marker_pulse = TablePT({'marker': [(0, 0), (duration_in_seconds_derived * 1e9, 0)]})
            full_template = AtomicMultiChannelPT(full_template, marker_pulse)
        finally:
            qupulse.utils.sympy.SYMPY_DURATION_ERROR_MARGIN = old_value

    def test_regression_sequencept_with_mappingpt(self):
        t1 = TablePT({'C1': [(0, 0), (100, 0)], 'C2': [(0, 1), (100, 1)]})
        t2 = ConstantPulseTemplate(200, {'C1': 2, 'C2': 3})
        qupulse_template = SequencePulseTemplate(t1, t2)
        channel_mapping = {'C1': None, 'C2': 'C2'}
        p = MappingPT(qupulse_template, channel_mapping=channel_mapping)
        plot(p)
        self.assertEqual(p.defined_channels, {'C2'})

    def test_expressions(self):
        cpt = ConstantPulseTemplate('duration', {'A': 5.4, 'B': 'amplitude_b'})
        self.assertEqual({'duration', 'amplitude_b'}, cpt.parameter_names)
        self.assertEqual(ExpressionScalar('duration'), cpt.duration)

        self.assertIsNone(cpt.build_waveform({'duration': 0., 'amplitude_b': 1.}, {'A': 'A', 'B': 'B'}))
        self.assertIsNone(cpt.build_waveform({'duration': 1., 'amplitude_b': 1.}, {'A': None, 'B': None}))

        wf1 = ConstantWaveform(duration=TimeType.from_float(1.4), channel='C', amplitude=1.6)
        wf2 = ConstantWaveform(duration=TimeType.from_float(1.5), channel='A', amplitude=5.4)
        self.assertEqual(wf1, cpt.build_waveform({'duration': 1.4, 'amplitude_b': 1.6}, {'A': None, 'B': 'C'}))
        self.assertEqual(wf2, cpt.build_waveform({'duration': 1.5, 'amplitude_b': None}, {'A': 'A', 'B': None}))

        wf3 = ConstantWaveform.from_mapping(duration=TimeType.from_float(1.6), constant_values={'C': 5.4, 'B': -.3})
        self.assertEqual(wf3, cpt.build_waveform({'duration': 1.6, 'amplitude_b': -.3}, {'A': 'C', 'B': 'B'}))

    def test_regression_defined_channels(self):
        p=ConstantPulseTemplate(100, {'a': 1.})
        q=ConstantPulseTemplate(100, {'b': 1.})
        pt=AtomicMultiChannelPulseTemplate(p, q)
        self.assertEqual(pt.defined_channels, {'a', 'b'})

    def test_build_waveform(self):
        tpt = ConstantPulseTemplate(200, {'C1': 2, 'C2': 3})

        wf_id = tpt.build_waveform({}, {'C1': 'C1', 'C2': 'C2'})
        self.assertEqual(
            ConstantWaveform.from_mapping(200, {'C1': 2, 'C2': 3}),
            wf_id
        )

        wf_1 = tpt.build_waveform({}, {'C1': 'C1', 'C2': None})
        self.assertEqual(
            ConstantWaveform.from_mapping(200, {'C1': 2}),
            wf_1
        )

        wf_2 = tpt.build_waveform({}, {'C1': None, 'C2': 'A'})
        self.assertEqual(
            ConstantWaveform.from_mapping(200, {'A': 3}),
            wf_2
        )

        wf_all = tpt.build_waveform({}, {'C1': 'B', 'C2': 'A'})
        self.assertEqual(
            ConstantWaveform.from_mapping(200, {'A': 3, 'B': 2}),
            wf_all
        )

        self.assertIsNone(tpt.build_waveform({}, {'C1': None, 'C2': None}))


class ConstantPulseTemplateSerializationTests(SerializableTests, unittest.TestCase):
    @property
    def class_to_test(self):
        return ConstantPulseTemplate

    def make_kwargs(self):
        return {
            'name': 'yoho',
            'duration': 'dur',
            'amplitude_dict': {'int': 1, 'float': -3.4, 'expr': 'x + y'},
            'measurements': [('m', 1, 1), ('foo', 'z', 'o')],
        }

    def assert_equal_instance_except_id(self, lhs: ConstantPulseTemplate, rhs: ConstantPulseTemplate):
        self.assertIsInstance(lhs, ConstantPulseTemplate)
        self.assertIsInstance(rhs, ConstantPulseTemplate)
        self.assertEqual(lhs._name, rhs._name)
        self.assertEqual(lhs.measurement_declarations, rhs.measurement_declarations)
        self.assertEqual(lhs._amplitude_dict, rhs._amplitude_dict)
        self.assertEqual(lhs.duration, rhs.duration)

    def test_legacy_deserialization(self):
        serialized = """{
            "#amplitudes": {
                "ZI0_A_MARKER_FRONT": 1
            },
            "#type": "qupulse.pulses.constant_pulse_template.ConstantPulseTemplate",
            "duration": 62848.0,
            "name": "constant_pulse"
        }"""
        backend = DictBackend()
        backend.storage['my_pt'] = serialized

        ps = PulseStorage(backend)

        deserialized = ps['my_pt']
        expected = ConstantPulseTemplate(
            amplitude_dict={"ZI0_A_MARKER_FRONT": 1},
            duration=62848, name="constant_pulse"
        )
        self.assert_equal_instance(expected, deserialized)
