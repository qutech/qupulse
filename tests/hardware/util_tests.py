import unittest
import itertools

import pytabor
import numpy as np

from qctoolkit.hardware.awgs.tabor import TaborSegment
from qctoolkit.hardware.util import voltage_to_uint16, make_combined_wave


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


class TaborMakeCombinedTest(unittest.TestCase):

    def exec_general(self, data_1, data_2):
        tabor_segments = [TaborSegment(d1, d2) for d1, d2 in zip(data_1, data_2)]
        expected_length = (sum(segment.num_points for segment in tabor_segments) + 16 * (len(tabor_segments) - 1)) * 2

        offset = 0
        pyte_result = 15000*np.ones(expected_length, dtype=np.uint16)
        for i, segment in enumerate(tabor_segments):
            offset = pytabor.make_combined_wave(segment[0], segment[1],
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


    def test_make_comb_both(self):
        gen = itertools.count()
        data_1 = [np.fromiter(gen, count=32, dtype=np.uint16),
                  np.fromiter(gen, count=16, dtype=np.uint16),
                  np.fromiter(gen, count=192, dtype=np.uint16)]

        data_2 = [np.fromiter(gen, count=32, dtype=np.uint16),
                  np.fromiter(gen, count=16, dtype=np.uint16),
                  np.fromiter(gen, count=192, dtype=np.uint16)]

        self.exec_general(data_1, data_2)

    def test_make_single_chan(self):
        gen = itertools.count()
        data_1 = [np.fromiter(gen, count=32, dtype=np.uint16),
                  np.fromiter(gen, count=16, dtype=np.uint16),
                  np.fromiter(gen, count=192, dtype=np.uint16)]

        data_2 = [None]*len(data_1)
        self.exec_general(data_1, data_2)
        self.exec_general(data_2, data_1)

