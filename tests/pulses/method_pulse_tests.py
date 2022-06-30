import unittest
import numpy as np

from qupulse.pulses.method_pulse_template import MethodPulseTemplate
from qupulse.expressions import ExpressionScalar
from qupulse.pulses.plotting import render



class MethodPulseTest(unittest.TestCase):
    def setUp(self) -> None:
        pulse_method=lambda t: np.sin(.2*t)
        self.fpt = MethodPulseTemplate(pulse_method, duration=100, channel='A')


class MethodPulsePropertyTest(MethodPulseTest):

    def test_defined_channels(self) -> None:
        self.assertEqual({'A'}, self.fpt.defined_channels)

    def test_parameter_names(self):
        self.assertEqual(self.fpt.parameter_names, set())

    def test_duration(self):
        self.assertEqual(self.fpt.duration, 100)

    def test_integral(self) -> None:
        pulse = MethodPulseTemplate(pulse_method=lambda t: 0*t, duration=30)
        self.assertDictEqual(pulse.integral, {'default': 0})
        pulse = MethodPulseTemplate(pulse_method=lambda t: 1+0*t, duration=30)
        self.assertDictEqual(pulse.integral, {'default': 30})
        pulse = MethodPulseTemplate(pulse_method=lambda t: np.sin(t), duration=30)
        self.assertDictEqual(pulse.integral, {'default': ExpressionScalar(0.8457485501124153)})
        

class MethodPulseSequencingTest(MethodPulseTest):
    def test_build_waveform(self) -> None:
        wf = self.fpt.build_waveform({}, channel_mapping={'A': 'B'})
        self.assertEqual(wf.defined_channels, {'B'})
        
    def test_sample(self) -> None:
        times, values, _ = render(self.fpt.create_program(), sample_rate = 2)
        np.testing.assert_almost_equal(values['A'], np.sin(.2 * np.arange(0,100.1, .5)))


if __name__=='__main__':
    unittest.main()
    