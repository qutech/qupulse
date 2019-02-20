import unittest
import math

from qupulse.hardware.awgs.virtual import VirtualAWG
from qupulse._program._loop import Loop


class VirtualAWGTests(unittest.TestCase):
    def test_init(self):
        vawg = VirtualAWG('asd', 5)

        self.assertEqual(vawg.identifier, 'asd')
        self.assertEqual(vawg.num_channels, 5)

    def test_no_markers(self):
        vawg = VirtualAWG('asd', 5)

        self.assertEqual(vawg.num_markers, 0)

    def test_sample_rate(self):
        vawg = VirtualAWG('asd', 5)

        self.assertTrue(math.isnan(vawg.sample_rate))
