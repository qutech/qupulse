import unittest
import inspect

import numpy as np

from qupulse.utils.types import (HashableNumpyArray, Collection, SequenceProxy, _FrozenDictByWrapping,
                                 _FrozenDictByInheritance)


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


class FrozenDictTests(unittest.TestCase):
    FrozenDictType = _FrozenDictByWrapping

    """This class can test general non mutable mappings"""
    def setUp(self) -> None:
        self.d = {'a': 1, 'b': 2}
        self.f = self.FrozenDictType(self.d)
        self.prev_state = dict(self.f)

    def tearDown(self) -> None:
        self.assertEqual(self.prev_state, dict(self.f))

    def test_init(self):
        d = {'a': 1, 'b': 2}

        f1 = self.FrozenDictType(d)
        f2 = self.FrozenDictType(**d)
        f3 = self.FrozenDictType(d.items())

        self.assertEqual(d, f1)
        self.assertEqual(d, f2)
        self.assertEqual(d, f3)

        self.assertEqual(d.keys(), f1.keys())
        self.assertEqual(d.keys(), f2.keys())
        self.assertEqual(d.keys(), f3.keys())

        self.assertEqual(set(d.items()), set(f1.items()))
        self.assertEqual(set(d.items()), set(f2.items()))
        self.assertEqual(set(d.items()), set(f3.items()))

    def test_mapping(self):
        d = {'a': 1, 'b': 2}
        f = self.FrozenDictType(d)

        self.assertEqual(len(d), len(f))
        self.assertIn('a', f)
        self.assertIn('b', f)
        self.assertNotIn('c', f)

        self.assertEqual(1, f['a'])
        self.assertEqual(2, f['b'])

        with self.assertRaisesRegex(KeyError, 'c'):
            _ = f['c']

        with self.assertRaises(TypeError):
            f['a'] = 9

        with self.assertRaises(TypeError):
            del f['a']

    def test_copy(self):
        d = {'a': 1, 'b': 2}
        f = self.FrozenDictType(d)
        self.assertIs(f, f.copy())

    def test_eq_and_hash(self):
        d = {'a': 1, 'b': 2}

        f1 = self.FrozenDictType(d)
        f2 = self.FrozenDictType({'a': 1, 'b': 2})
        f3 = self.FrozenDictType({'a': 1, 'c': 3})

        self.assertEqual(f1, f2)
        self.assertEqual(hash(f1), hash(f2))

        self.assertNotEqual(f1, f3)


class FrozenDictByInheritanceTests(FrozenDictTests):
    FrozenDictType = _FrozenDictByInheritance

    def test_update(self):
        with self.assertRaisesRegex(TypeError, 'immutable'):
            self.f.update(d=5)

    def test_setdefault(self):
        with self.assertRaisesRegex(TypeError, 'immutable'):
            self.f.setdefault('c', 3)
        with self.assertRaisesRegex(TypeError, 'immutable'):
            self.f.setdefault('a', 2)

    def test_clear(self):
        with self.assertRaisesRegex(TypeError, 'immutable'):
            self.f.clear()

    def test_pop(self):
        with self.assertRaisesRegex(TypeError, 'immutable'):
            self.f.pop()
        with self.assertRaisesRegex(TypeError, 'immutable'):
            self.f.pop('a')

    def test_popitem(self):
        with self.assertRaisesRegex(TypeError, 'immutable'):
            self.f.popitem()
