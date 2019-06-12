import unittest

import numpy as np

from qupulse.utils.types import HashableNumpyArray


class HashableNumpyArrayTest(unittest.TestCase):
    def test_hash(self):

        a = np.array([1, 2, 3]).view(HashableNumpyArray)

        b = np.array([3, 4, 5]).view(HashableNumpyArray)

        c = np.array([1, 2, 3]).view(HashableNumpyArray)

        self.assertNotEqual(hash(a), hash(b))
        self.assertEqual(hash(a), hash(c))
