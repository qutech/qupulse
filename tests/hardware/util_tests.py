import unittest
import numpy as np

from qctoolkit.hardware.util import voltage_to_uint16


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
