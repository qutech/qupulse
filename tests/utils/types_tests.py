import unittest

import numpy as np

from qupulse.utils.types import HashableNumpyArray, Collection


class HashableNumpyArrayTest(unittest.TestCase):
    def test_hash(self):

        a = np.array([1, 2, 3]).view(HashableNumpyArray)

        b = np.array([3, 4, 5]).view(HashableNumpyArray)

        c = np.array([1, 2, 3]).view(HashableNumpyArray)

        self.assertNotEqual(hash(a), hash(b))
        self.assertEqual(hash(a), hash(c))


class CollectionTests(unittest.TestCase):
    def test_isinstance(self):
        self.assertTrue(isinstance(set(), Collection))
        self.assertTrue(isinstance(list(), Collection))
        self.assertTrue(isinstance(tuple(), Collection))

        self.assertFalse(isinstance(5, Collection))

