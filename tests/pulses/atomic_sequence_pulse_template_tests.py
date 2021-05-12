import unittest
import numpy as np

from qupulse.pulses.constant_pulse_template import ConstantPulseTemplate
from qupulse.pulses.atomic_sequence_pulse_template import AtomicSequencePulseTemplate
from qupulse.pulses.sequence_pulse_template import SequencePulseTemplate
from qupulse.pulses.plotting import render


class TestAtomicSequencePulseTemplate(unittest.TestCase):

    def setUp(self):
        p1 = ConstantPulseTemplate(4, {'P1': 1.0})
        p2 = ConstantPulseTemplate(4, {'P1': 2.0})
        pt = AtomicSequencePulseTemplate(p1, p2,)
        self.atomic_spt = pt

    def test_AtomicSequencePulseTemplate_integral(self):
        pt = self.atomic_spt

        self.assertDictEqual(pt.integral, {'P1': 12.})

    def test_AtomicSequencePulseTemplate_render(self):
        pt = self.atomic_spt
        times, data, measurements = render(pt.create_program(), sample_rate=1)
        np.testing.assert_array_almost_equal(times, np.arange(0, 9.))
        self.assertEqual(set(data.keys()), set(['P1']))

    def test_AtomicSequencePulseTemplate_measurements(self):

        p1 = ConstantPulseTemplate(10, {'P1': 1.0}, measurements=[('a', 0, 2)])
        p2 = ConstantPulseTemplate(10, {'P1': 2.0}, measurements=[('a', 0, 2), ('b', 0, 5)])

        pt = SequencePulseTemplate(p1, p2, measurements=[('c', 0, 3)])

        measurement_windows0 = pt.create_program().get_measurement_windows()

        pt = AtomicSequencePulseTemplate(p1, p2, measurements=[('c', 0, 3)])
        measurement_windows = pt.create_program().get_measurement_windows()

        self.assertEqual(measurement_windows0.keys(), measurement_windows.keys())
        for key in measurement_windows0:
            np.testing.assert_array_equal(measurement_windows0[key], measurement_windows[key])

    def test_AtomicSequencePulseTemplate_measurement_names(self):
        p1 = ConstantPulseTemplate(10, {'P1': 1.0})
        p2 = ConstantPulseTemplate(10, {'P1': 2.0})
        pt = AtomicSequencePulseTemplate(p1, p2,)
        self.assertEqual(pt.measurement_names, set())
        pt = AtomicSequencePulseTemplate(p1, p2, measurements=[('c', 0, 3)])
        self.assertEqual(pt.measurement_names, {'c'})


if __name__ == "__main__":
    unittest.main(verbosity=2)
