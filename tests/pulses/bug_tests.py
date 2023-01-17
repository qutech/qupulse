import unittest
from unittest import mock

import numpy as np

from qupulse.pulses.table_pulse_template import TablePulseTemplate
from qupulse.pulses.function_pulse_template import FunctionPulseTemplate
from qupulse.pulses.sequence_pulse_template import SequencePulseTemplate
from qupulse.pulses.repetition_pulse_template import RepetitionPulseTemplate
from qupulse.pulses.multi_channel_pulse_template import AtomicMultiChannelPulseTemplate
from qupulse.pulses.mapping_pulse_template import MappingPulseTemplate
from qupulse.pulses.loop_pulse_template import ForLoopPulseTemplate

from qupulse.plotting import plot

from qupulse._program._loop import to_waveform
from qupulse.utils import isclose

class BugTests(unittest.TestCase):

    def test_plotting_two_channel_function_pulse_after_two_channel_table_pulse_crash(self) -> None:
        """ successful if no crash -> no asserts """
        template = TablePulseTemplate(entries={'A': [(0, 0),
                                          ('ta', 'va', 'hold'),
                                          ('tb', 'vb', 'linear'),
                                          ('tend', 0, 'jump')],
                                    'B': [(0, 0),
                                          ('ta', '-va', 'hold'),
                                          ('tb', '-vb', 'linear'),
                                          ('tend', 0, 'jump')]}, measurements=[('m', 0, 'ta'),
                                                                               ('n', 'tb', 'tend-tb')])

        parameters = {'ta': 2,
                      'va': 2,
                      'tb': 4,
                      'vb': 3,
                      'tc': 5,
                      'td': 11,
                      'tend': 6}
        _ = plot(template, parameters, sample_rate=100, show=False, plot_measurements={'m', 'n'})

        repeated_template = RepetitionPulseTemplate(template, 'n_rep')
        sine_template = FunctionPulseTemplate('sin_a*sin(t)', '2*3.1415')
        two_channel_sine_template = AtomicMultiChannelPulseTemplate(
            (sine_template, {'default': 'A'}),
            (sine_template, {'default': 'B'}, {'sin_a': 'sin_b'})
        )
        sequence_template = SequencePulseTemplate(repeated_template, two_channel_sine_template)
        #sequence_template = SequencePulseTemplate(two_channel_sine_template, repeated_template) # this was working fine

        sequence_parameters = dict(parameters)  # we just copy our parameter dict from before
        sequence_parameters['n_rep'] = 4  # and add a few new values for the new params from the sine wave
        sequence_parameters['sin_a'] = 1
        sequence_parameters['sin_b'] = 2

        _ = plot(sequence_template, parameters=sequence_parameters, sample_rate=100, show=False)

    def test_plot_with_parameter_value_being_expression_string(self) -> None:
        sine_measurements = [('M', 't_duration/2', 't_duration')]
        sine = FunctionPulseTemplate('a*sin(omega*t)', 't_duration', measurements=sine_measurements)
        sine_channel_mapping = dict(default='sin_channel')
        sine_measurement_mapping = dict(M='M_sin')
        remapped_sine = MappingPulseTemplate(sine, measurement_mapping=sine_measurement_mapping,
                                             channel_mapping=sine_channel_mapping)
        cos_measurements = [('M', 0, 't_duration/2')]
        cos = FunctionPulseTemplate('a*cos(omega*t)', 't_duration', measurements=cos_measurements)
        cos_channel_mapping = dict(default='cos_channel')
        cos_measurement_mapping = dict(M='M_cos')
        remapped_cos = MappingPulseTemplate(cos, channel_mapping=cos_channel_mapping, measurement_mapping=cos_measurement_mapping)
        both = AtomicMultiChannelPulseTemplate(remapped_sine, remapped_cos)

        parameter_values = dict(omega=1.0, a=1.0, t_duration="2*pi")

        _ = plot(both, parameters=parameter_values, sample_rate=100)

    def test_issue_584_uninitialized_table_sample(self):
        """issue 584"""
        d = 598.3333333333334 - 480
        tpt = TablePulseTemplate(entries={'P': [(0, 1.0, 'hold'), (d, 1.0, 'hold')]})
        with mock.patch('qupulse._program.waveforms.PULSE_TO_WAVEFORM_ERROR', 1e-6):
            wf = to_waveform(tpt.create_program())
            self.assertTrue(isclose(d, wf.duration, abs_tol=1e-6))

            start_time = 0.
            end_time = wf.duration
            sample_rate = 3.

            sample_count = (end_time - start_time) * sample_rate + 1

            times = np.linspace(float(start_time), float(wf.duration), num=int(sample_count), dtype=float)
            times[-1] = np.nextafter(times[-1], times[-2])

            out = np.full_like(times, fill_value=np.nan)
            sampled = wf.get_sampled(channel='P', sample_times=times, output_array=out)

            expected = np.full_like(times, fill_value=1.)
            np.testing.assert_array_equal(expected, sampled)

    def test_issue_612_for_loop_duration(self):
        fpt = FunctionPulseTemplate('sin(2*pi*i*t*f)', '1/f')
        pt = ForLoopPulseTemplate(fpt, 'i', 'floor(total_time*f)')
        self.assertEqual(
            (500 + 501) // 2,
            pt.duration.evaluate_in_scope({'f': 1., 'total_time': 500})
        )

