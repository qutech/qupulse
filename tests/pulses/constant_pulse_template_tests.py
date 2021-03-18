import unittest

import qupulse.pulses.plotting
from qupulse.pulses import TablePT, FunctionPT, AtomicMultiChannelPT, MappingPT
from qupulse.pulses.plotting import render, plot
from qupulse.pulses.sequence_pulse_template import SequencePulseTemplate
from qupulse._program._loop import make_compatible

from qupulse.pulses.constant_pulse_template import ConstantPulseTemplate
#, AtomicSequencePulseTemplate, create_multichannel_pulse, MethodPulseTemplate

class TestConstantPulseTemplate(unittest.TestCase):

    def test_ConstantPulseTemplate(self):
        pt = ConstantPulseTemplate(100, {'P1': .5, 'P2': .25})
        self.assertEqual(pt.integral, {'P1': 50, 'P2': 25})

    def test_zero_duration(self):
        p1 = ConstantPulseTemplate(10, {'P1': 1.})
        p2 = ConstantPulseTemplate(0, {'P1': 1.})
        p3 = ConstantPulseTemplate(2, {'P1': 1.})

        _ = qupulse.pulses.plotting.render(p1.create_program())

        pulse = AtomicSequencePulseTemplate([p1, p2, p3])
        prog = pulse.create_program()
        _ = qupulse.pulses.plotting.render(prog)

        self.assertEqual(pulse.duration, 12)

    def test_regression_duration_conversion(self):
        for duration_in_samples in [64, 936320, 24615392]:
            p = ConstantPulseTemplate(duration_in_samples / 2.4, {'a': 0})
            number_of_samples = p.create_program().duration * 2.4
            make_compatible(p.create_program(), 8, 8, 2.4)
            self.assertEqual(number_of_samples.denominator, 1)

            p2 = ConstantPulseTemplate((duration_in_samples +1) / 2.4, {'a': 0})
            self.assertNotEqual(p.create_program().duration, p2.create_program().duration)

    def test_regression_duration_conversion_functionpt(self):
        for duration_in_samples in [64, 2000, 936320]:
            p = FunctionPT('1', duration_expression=duration_in_samples / 2.4, channel='a')
            number_of_samples = p.create_program().duration * 2.4
            self.assertEqual(number_of_samples.denominator, 1)

    def test_regression_template_combination(self):
        duration_in_seconds = 2e-6
        full_template = ConstantPulseTemplate(duration=duration_in_seconds * 1e9, amplitude_dict={'C1': 1.1})
        duration_in_seconds_derived = 1e-9 * full_template.duration
        marker_pulse = TablePT({'marker': [(0, 0), (duration_in_seconds_derived * 1e9, 0)]})
        full_template = AtomicMultiChannelPT(full_template, marker_pulse)

    def test_regression_sequencept_with_mappingpt(self):
        t1 = TablePT({'C1': [(0, 0), (100, 0)], 'C2': [(0, 1), (100, 1)]})
        t2 = ConstantPulseTemplate(200, {'C1': 2, 'C2': 3})
        qupulse_template = SequencePulseTemplate(t1, t2)
        channel_mapping = {'C1': None, 'C2': 'C2'}
        p = MappingPT(qupulse_template, channel_mapping=channel_mapping)
        plot(p)
        self.assertEqual(p.defined_channels, {'C2'})
