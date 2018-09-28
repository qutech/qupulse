import unittest
import itertools

import pytabor

import numpy as np

from qupulse.hardware.awgs.tabor import TaborSegment
from qupulse.hardware.util import voltage_to_uint16, make_combined_wave, find_positions


from . import dummy_modules

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

def validate_result(tabor_segments, result, fill_value=None):
    pos = 0
    for i, tabor_segment in enumerate(tabor_segments):
        if i > 0:
            if tabor_segment.ch_b is None:
                if fill_value:
                    np.testing.assert_equal(result[pos:pos + 16], np.full(16, fill_value=fill_value, dtype=np.uint16))
            else:
                np.testing.assert_equal(result[pos:pos + 16], np.full(16, tabor_segment.ch_b[0], dtype=np.uint16))
            pos += 16

            if tabor_segment.ch_a is None:
                if fill_value:
                    np.testing.assert_equal(result[pos:pos + 16], np.full(16, fill_value=fill_value, dtype=np.uint16))
            else:
                np.testing.assert_equal(result[pos:pos + 16], np.full(16, tabor_segment.ch_a[0], dtype=np.uint16))
            pos += 16

        for j in range(tabor_segment.num_points // 16):
            if tabor_segment.ch_b is None:
                if fill_value:
                    np.testing.assert_equal(result[pos:pos + 16], np.full(16, fill_value=fill_value, dtype=np.uint16))
            else:
                np.testing.assert_equal(result[pos:pos + 16], tabor_segment.ch_b[j * 16: (j + 1) * 16])
            pos += 16

            if tabor_segment.ch_a is None:
                if fill_value:
                    np.testing.assert_equal(result[pos:pos + 16], np.full(16, fill_value=fill_value, dtype=np.uint16))
            else:
                np.testing.assert_equal(result[pos:pos + 16], tabor_segment.ch_a[j * 16: (j + 1) * 16])
            pos += 16


class TaborMakeCombinedTest(unittest.TestCase):
    def exec_general(self, data_1, data_2, fill_value=None):
        tabor_segments = [TaborSegment(d1, d2, None, None) for d1, d2 in zip(data_1, data_2)]
        expected_length = (sum(segment.num_points for segment in tabor_segments) + 16 * (len(tabor_segments) - 1)) * 2

        result = make_combined_wave(tabor_segments, fill_value=fill_value)
        self.assertEqual(len(result), expected_length)

        validate_result(tabor_segments, result, fill_value=fill_value)

        destination_array = np.empty(expected_length, dtype=np.uint16)
        result = make_combined_wave(tabor_segments, fill_value=fill_value, destination_array=destination_array)
        validate_result(tabor_segments, result, fill_value=fill_value)
        self.assertEqual(destination_array.__array_interface__['data'], result.__array_interface__['data'],
                         'Data was copied')

    def test_make_comb_both(self):
        gen = itertools.count()
        data_1 = [np.fromiter(gen, count=32, dtype=np.uint16),
                  np.fromiter(gen, count=16, dtype=np.uint16),
                  np.fromiter(gen, count=192, dtype=np.uint16)]

        data_2 = [np.fromiter(gen, count=32, dtype=np.uint16),
                  np.fromiter(gen, count=16, dtype=np.uint16),
                  np.fromiter(gen, count=192, dtype=np.uint16)]
        for d in data_2:
            d += 1000

        self.exec_general(data_1, data_2)
        self.exec_general(data_1, data_2, 2000)

    def test_make_single_chan(self):
        gen = itertools.count()
        data_1 = [np.fromiter(gen, count=32, dtype=np.uint16),
                  np.fromiter(gen, count=16, dtype=np.uint16),
                  np.fromiter(gen, count=192, dtype=np.uint16)]

        data_2 = [None]*len(data_1)
        self.exec_general(data_1, data_2)
        self.exec_general(data_2, data_1)

        self.exec_general(data_1, data_2, 2000)
        self.exec_general(data_2, data_1, 2000)

    def test_empty_segment_list(self):
        combined = make_combined_wave([])

        self.assertIsInstance(combined, np.ndarray)
        self.assertIs(combined.dtype, np.dtype('uint16'))
        self.assertEqual(len(combined), 0)

    def test_invalid_segment_length(self):
        gen = itertools.count()
        data_1 = [np.fromiter(gen, count=32, dtype=np.uint16),
                  np.fromiter(gen, count=15, dtype=np.uint16),
                  np.fromiter(gen, count=192, dtype=np.uint16)]

        data_2 = [np.fromiter(gen, count=32, dtype=np.uint16),
                  np.fromiter(gen, count=16, dtype=np.uint16),
                  np.fromiter(gen, count=193, dtype=np.uint16)]

        tabor_segments = [TaborSegment(d, d, None, None) for d in data_1]
        with self.assertRaises(ValueError):
            make_combined_wave(tabor_segments)

        tabor_segments = [TaborSegment(d, d, None, None) for d in data_2]
        with self.assertRaises(ValueError):
            make_combined_wave(tabor_segments)


@unittest.skipIf(pytabor is dummy_modules.dummy_pytabor, "Cannot compare to pytabor results")
class TaborMakeCombinedPyTaborCompareTest(TaborMakeCombinedTest):
    def exec_general(self, data_1, data_2, fill_value=None):
        tabor_segments = [TaborSegment(d1, d2, None, None) for d1, d2 in zip(data_1, data_2)]
        expected_length = (sum(segment.num_points for segment in tabor_segments) + 16 * (len(tabor_segments) - 1)) * 2

        offset = 0
        pyte_result = 15000*np.ones(expected_length, dtype=np.uint16)
        for i, segment in enumerate(tabor_segments):
            offset = pytabor.make_combined_wave(segment.ch_a, segment.ch_b,
                                                dest_array=pyte_result, dest_array_offset=offset,
                                                add_idle_pts=i > 0)
        self.assertEqual(expected_length, offset)

        result = make_combined_wave(tabor_segments, fill_value=15000)
        np.testing.assert_equal(pyte_result, result)

        dest_array = 15000*np.ones(expected_length, dtype=np.uint16)
        result = make_combined_wave(tabor_segments, destination_array=dest_array)
        np.testing.assert_equal(pyte_result, result)
        # test that the destination array data is not copied
        self.assertEqual(dest_array.__array_interface__['data'],
                         result.__array_interface__['data'])

        with self.assertRaises(ValueError):
            make_combined_wave(tabor_segments, destination_array=np.ones(16))


class FindPositionTest(unittest.TestCase):
    def test_find_position(self):
        data = [2, 6, -24, 65, 46, 5, -10, 9]
        to_find = [54, 12, 5, -10, 45, 6, 2]

        positions = find_positions(data, to_find)

        self.assertEqual(positions.tolist(), [-1, -1, 5, 6, -1, 1, 0])
