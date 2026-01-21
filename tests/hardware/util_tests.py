import unittest

import numpy as np

try:
    import zhinst.utils
except ImportError:
    zhinst = None

from qupulse.utils.types import TimeType
from qupulse.hardware.util import voltage_to_uint16, find_positions, get_sample_times, not_none_indices, \
    zhinst_voltage_to_uint16
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

    def test_pow_2_divisor(self):
        sample_rate = TimeType.from_fraction(12, 5)
        wf = DummyWaveform(duration=TimeType.from_fraction(400, 12))
        
        wf._pow_2_divisor = 3
        times, n_samples = get_sample_times(wf, sample_rate_in_GHz=sample_rate)
        
        # the expected times are still at original sample rate, just with less
        # max values, as the logic of having one time-array
        # for all waveforms (which assumes a fixed sample rate)
        # would not allow intercepting those here.
        expected_times = np.arange(10) / float(sample_rate)
        np.testing.assert_almost_equal(times, expected_times, decimal=10)
        
        #the segment length however comes back reduced, 10 instead of 80
        expected_len = np.asarray(10)
        np.testing.assert_equal(n_samples, expected_len)
        
        
class NotNoneIndexTest(unittest.TestCase):
    def test_not_none_indices(self):
        self.assertEqual(([None, 0, 1, None, None, 2], 3),
                         not_none_indices([None, 'a', 'b', None, None, 'c']))


@unittest.skipIf(zhinst is None, "zhinst not installed")
class ZHInstVoltageToUint16Test(unittest.TestCase):
    def test_size_exception(self):
        with self.assertRaisesRegex(ValueError, "No input"):
            zhinst_voltage_to_uint16(None, None, (None, None, None, None))
        with self.assertRaisesRegex(ValueError, "dimension"):
            zhinst_voltage_to_uint16(np.zeros(192), np.zeros(191), (None, None, None, None))
        with self.assertRaisesRegex(ValueError, "dimension"):
            zhinst_voltage_to_uint16(np.zeros(192), None, (np.zeros(191), None, None, None))

    def test_range_exception(self):
        with self.assertRaisesRegex(ValueError, "invalid"):
            zhinst_voltage_to_uint16(2.*np.ones(192), None, (None, None, None, None))
        with self.assertRaisesRegex(ValueError, "invalid"):
            zhinst_voltage_to_uint16(None, 2.*np.ones(192), (None, None, None, None))
        # this should work
        zhinst_voltage_to_uint16(None, None, (2. * np.ones(192), None, None, None))

    def test_zeros(self):
        combined = zhinst_voltage_to_uint16(None, np.zeros(192), (None, None, None, None))
        np.testing.assert_array_equal(np.zeros(3*192, dtype=np.uint16), combined)

    def test_full(self):
        ch1 = np.linspace(0, 1., num=192)
        ch2 = np.linspace(0., -1., num=192)

        markers = tuple(np.array(([1.] + [0.]*m) * 192)[:192] for m in range(1, 5))

        combined = zhinst_voltage_to_uint16(ch1, ch2, markers)

        marker_data = [sum(int(markers[m][idx] > 0) << m for m in range(4))
                       for idx in range(192)]
        marker_data = np.array(marker_data, dtype=np.uint16)
        expected = zhinst.utils.convert_awg_waveform(ch1, ch2, marker_data)

        np.testing.assert_array_equal(expected, combined)
