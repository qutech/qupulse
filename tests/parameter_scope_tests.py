import unittest
from unittest import mock

from qupulse.parameter_scope import Scope, DictScope, MappedScope, ParameterNotProvidedException, NonVolatileChange

from qupulse.utils.types import FrozenDict


class DictScopeTests(unittest.TestCase):
    def test_init(self):
        with self.assertRaises(AssertionError):
            DictScope(dict())
        fd = FrozenDict({'a': 2})
        ds = DictScope(fd)
        self.assertIs(fd, ds._values)
        self.assertIs(frozenset(), ds._volatile_parameters)

        vp = frozenset('a')
        ds = DictScope(fd, vp)
        self.assertIs(fd, ds._values)
        self.assertIs(vp, ds._volatile_parameters)

    def test_mapping(self):
        ds = DictScope(FrozenDict({'a': 1, 'b': 2}))
        self.assertIn('a', ds)
        self.assertNotIn('c', ds)

        self.assertEqual(set('ab'), set(ds.keys()))
        self.assertEqual(set('ab'), set(ds))
        self.assertEqual({1, 2}, set(ds.values()))
        self.assertEqual({('a', 1), ('b', 2)}, set(ds.items()))
        self.assertEqual({'a': 1, 'b': 2}, dict(ds))

        self.assertEqual(1, ds['a'])

        with self.assertRaises(KeyError):
            ds['c']

        with self.assertRaises(TypeError):
            ds['a'] = 3

    def test_change_constants(self):
        volatile = frozenset('b')
        ds = DictScope(FrozenDict({'a': 1, 'b': 2}), volatile=volatile)

        changes = {'b': 3, 'c': 4}
        ds2 = ds.change_constants(changes)
        self.assertEqual({'a': 1, 'b': 2}, dict(ds))
        self.assertEqual(volatile, ds.get_volatile_parameters())
        self.assertEqual({'a': 1, 'b': 3}, dict(ds2))
        self.assertEqual(volatile, ds2.get_volatile_parameters())

        with self.assertWarns(NonVolatileChange):
            ds.change_constants({'a': 2, 'b': 3, 'c': 4})

    def test_get_parameter(self):
        ds = DictScope(FrozenDict({'a': 1, 'b': 2}))

        self.assertEqual(1, ds.get_parameter('a'))
        self.assertEqual(2, ds.get_parameter('b'))

        with self.assertRaises(ParameterNotProvidedException) as cm:
            ds.get_parameter('c')
        self.assertEqual('c', cm.exception.parameter_name)

    def test_get_volatile(self):
        volatile = frozenset('ab')
        ds = DictScope(FrozenDict({'a': 1, 'b': 2}), volatile=volatile)
        self.assertEqual(volatile, ds.get_volatile_parameters())

    def test_eq(self):
        ds = DictScope(FrozenDict({'a': 1, 'b': 2}), volatile=frozenset('a'))

        ds1 = DictScope(FrozenDict({'a': 1, 'b': 2}), volatile=frozenset())
        ds2 = DictScope(FrozenDict({'a': 1, 'b': 2}), volatile=frozenset('a'))
        ds3 = DictScope(FrozenDict({'a': 1, 'b': 2}), volatile=frozenset('ab'))
        ds4 = DictScope(FrozenDict({'a': 1, 'b': 2, 'c': 3}), volatile=frozenset('a'))

        self.assertNotEqual(ds, ds1)
        self.assertNotEqual(ds, ds3)
        self.assertNotEqual(ds, ds4)
        self.assertEqual(ds, ds2)
        self.assertEqual(hash(ds), hash(ds2))

        self.assertNotEqual(ds1, ds2)
        self.assertNotEqual(ds2, ds3)
        self.assertNotEqual(ds3, ds4)

    def test_from_mapping(self):
        m = {'a': 1, 'b': 2}
        volatile = {'a'}
        ds = DictScope.from_mapping(m.copy())
        self.assertEqual(m, dict(ds))
        self.assertEqual(frozenset(), ds.get_volatile_parameters())

        ds = DictScope.from_mapping(m.copy(), volatile=volatile.copy())
        self.assertEqual(DictScope(FrozenDict(m), frozenset(volatile)), ds)

    def test_from_kwargs(self):
        m = {'a': 1, 'b': 2}
        volatile = {'a'}
        ds = DictScope.from_kwargs(a=1, b=2)
        self.assertEqual(DictScope(FrozenDict(m)), ds)

        ds = DictScope.from_kwargs(a=1, b=2, volatile=volatile.copy())
        self.assertEqual(DictScope(FrozenDict(m), frozenset(volatile)), ds)


class MappedScopeTests(unittest.TestCase):
    def test_init(self):
        raise NotImplementedError()
