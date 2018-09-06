import unittest

from qupulse.pulses.table_pulse_template import TablePulseTemplate
from qupulse.pulses.function_pulse_template import FunctionPulseTemplate
from qupulse.pulses.sequence_pulse_template import SequencePulseTemplate
from qupulse.pulses.repetition_pulse_template import RepetitionPulseTemplate
from qupulse.pulses.multi_channel_pulse_template import AtomicMultiChannelPulseTemplate

from qupulse.pulses.plotting import plot


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
