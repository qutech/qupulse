import unittest

from qctoolkit.pulses.measurements import Measurement


class MeasurementTest(unittest.TestCase):
    def setUp(self):
        self.m = Measurement(10)
        self.m.measure(5)
        self.m.measure(5, 2)

    def test_toList(self):
        self.assertEqual([x for x in self.m], [(0, 5), (7, 10)])