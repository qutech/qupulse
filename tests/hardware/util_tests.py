import unittest

import numpy as np

from qupulse.utils.types import TimeType
from qupulse.hardware.util import voltage_to_uint16, find_positions, get_sample_times
from tests.pulses.sequencing_dummies import DummyWaveform


class VoltageToBinaryTests(unittest.TestCase):

    def test_voltage_to_uint16(self):

        with self.assertRaises(ValueError):
            voltage_to_uint16(np.zeros(0), 0, 0, 0)

        linspace_voltage = np.linspace(0, 1, 128)
        with self.assertRaises(ValueError):
            voltage_to_uint16(linspace_voltage, 0.9, 0, 1)

        with self.assertRaises(ValueError):
            voltage_to_uint16(linspace_voltage, 1.1, -1, 1)

        expected_data = np.arange(0, 128, dtype=np.uint16)
        received_data = voltage_to_uint16(linspace_voltage, 0.5, 0.5, 7)

        self.assertTrue(np.all(expected_data == received_data))

    def test_zero_level_14bit(self):
        zero_level = voltage_to_uint16(np.zeros(1), 0.5, 0., 14)
        self.assertEqual(zero_level, 8192)




class FindPositionTest(unittest.TestCase):
    def test_find_position(self):
        data = [2, 6, -24, 65, 46, 5, -10, 9]
        to_find = [54, 12, 5, -10, 45, 6, 2]

        positions = find_positions(data, to_find)

        self.assertEqual(positions.tolist(), [-1, -1, 5, 6, -1, 1, 0])


class SampleTimeCalculationTest(unittest.TestCase):
    def test_get_sample_times(self):
        sample_rate = TimeType.from_fraction(12, 10)
        wf1 = DummyWaveform(duration=TimeType.from_fraction(20, 12))
        wf2 = DummyWaveform(duration=TimeType.from_fraction(400000000001, 120000000000))
        wf3 = DummyWaveform(duration=TimeType.from_fraction(1, 10**15))

        expected_times = np.arange(4) / 1.2
        times, n_samples = get_sample_times([wf1, wf2], sample_rate_in_GHz=sample_rate)
        np.testing.assert_equal(expected_times, times)
        np.testing.assert_equal(n_samples, np.asarray([2, 4]))

        with self.assertRaises(AssertionError):
            get_sample_times([], sample_rate_in_GHz=sample_rate)

        with self.assertRaisesRegex(ValueError, "non integer length"):
            get_sample_times([wf1, wf2], sample_rate_in_GHz=sample_rate, tolerance=0.)

        with self.assertRaisesRegex(ValueError, "length <= zero"):
            get_sample_times([wf1, wf3], sample_rate_in_GHz=sample_rate)

    def test_get_sample_times_single_wf(self):
        sample_rate = TimeType.from_fraction(12, 10)
        wf = DummyWaveform(duration=TimeType.from_fraction(40, 12))

        expected_times = np.arange(4) / 1.2
        times, n_samples = get_sample_times(wf, sample_rate_in_GHz=sample_rate)

        np.testing.assert_equal(times, expected_times)
        np.testing.assert_equal(n_samples, np.asarray(4))
