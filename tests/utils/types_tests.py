import unittest

import numpy as np

from qupulse.utils.types import HashableNumpyArray, Collection, SequenceProxy


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


class FrozenDict(unittest.TestCase):
    def test_init(self):
        raise NotImplementedError()

    def test_mapping(self):
        raise NotImplementedError()

    def test_eq_and_hash(self):
        raise NotImplementedError()
