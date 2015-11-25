import unittest

from qctoolkit.pulses.measurements import Measurement


class MeasurementTest(unittest.TestCase):
    def setUp(self):
        self.m = Measurement(10)
        self.m.measure(0, 5)
        self.m.measure(2, 5)
        self.inst = self.m.instantiate({})

    def test_toList(self):
        self.assertEqual([x for x in self.inst], [(0, 5), (7, 10)])