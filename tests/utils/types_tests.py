import unittest

import numpy as np

from qupulse.utils.types import (HashableNumpyArray, SequenceProxy,)


class HashableNumpyArrayTest(unittest.TestCase):
    def test_hash(self):

        with self.assertWarns(DeprecationWarning):
            a = np.array([1, 2, 3]).view(HashableNumpyArray)

            b = np.array([3, 4, 5]).view(HashableNumpyArray)

            c = np.array([1, 2, 3]).view(HashableNumpyArray)

        self.assertNotEqual(hash(a), hash(b))
        self.assertEqual(hash(a), hash(c))


class SequenceProxyTest(unittest.TestCase):
    def test_sequence_proxy(self):
        l = [1, 2, 3, 4, 1]
        p = SequenceProxy(l)
        self.assertEqual(l, list(iter(p)))
        self.assertEqual(5, len(p))
        self.assertEqual(3, p[2])
        self.assertEqual(list(reversed(l)), list(reversed(p)))
        self.assertEqual(2, p.index(3))
        self.assertEqual(2, p.count(1))
        self.assertIn(3, p)
        self.assertNotIn(5, p)

        with self.assertRaises(TypeError):
            p[1] = 7
