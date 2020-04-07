import unittest
from unittest import mock

from qupulse.parameter_scope import Scope, DictScope, MappedScope, ParameterNotProvidedException, NonVolatileChange
from qupulse.expressions import ExpressionScalar

from qupulse.utils.types import FrozenDict


class DictScopeTests(unittest.TestCase):
    def test_init(self):
        with self.assertRaises(AssertionError):
            DictScope(dict())
        fd = FrozenDict({'a': 2})
        ds = DictScope(fd)
        self.assertIs(fd, ds._values)
        self.assertEqual(FrozenDict(), ds._volatile_parameters)

        vp = frozenset('a')
        ds = DictScope(fd, vp)
        self.assertIs(fd, ds._values)
        self.assertEqual(FrozenDict(a=ExpressionScalar('a')), ds._volatile_parameters)

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
        volatile_dict = FrozenDict(b=ExpressionScalar('b'))
        ds = DictScope(FrozenDict({'a': 1, 'b': 2}), volatile=volatile)

        changes = {'b': 3, 'c': 4}
        ds2 = ds.change_constants(changes)
        self.assertEqual({'a': 1, 'b': 2}, dict(ds))
        self.assertEqual(volatile_dict, ds.get_volatile_parameters())
        self.assertEqual({'a': 1, 'b': 3}, dict(ds2))
        self.assertEqual(volatile_dict, ds2.get_volatile_parameters())

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
        volatile_dict = FrozenDict(a=ExpressionScalar('a'), b=ExpressionScalar('b'))
        ds = DictScope(FrozenDict({'a': 1, 'b': 2}), volatile=volatile)
        self.assertEqual(volatile_dict, ds.get_volatile_parameters())

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
        self.assertEqual(FrozenDict(), ds.get_volatile_parameters())

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
    def test_mapping(self):
        ds = DictScope.from_kwargs(a=1, b=2, c=3)
        ms = MappedScope(ds, FrozenDict(x=ExpressionScalar('a * b'),
                                        c=ExpressionScalar('a - b')))

        self.assertEqual(4, len(ms))
        self.assertEqual(set('abcx'), set(ms.keys()))
        self.assertEqual([-1, 1, 2, 2], sorted(ms.values()))
        self.assertEqual({('a', 1), ('b', 2), ('c', -1), ('x', 2)}, set(ms.items()))
        self.assertEqual(set(ms), set(ms.keys()))

        self.assertIn('a', ms)
        self.assertIn('c', ms)
        self.assertIn('x', ms)

        self.assertNotIn('d', ms)
        self.assertEqual(-1, ms['c'])
        self.assertEqual(1, ms['a'])
        self.assertEqual(2, ms['x'])

        with self.assertRaises(TypeError):
            ms['d'] = 9

        with self.assertRaisesRegex(KeyError, 'd'):
            _ = ms['d']

    def test_parameter(self):
        mock_a = mock.Mock(wraps=1)
        mock_result = mock.Mock()

        ds = DictScope.from_kwargs(a=mock_a, b=2, c=3)
        ms = MappedScope(ds, FrozenDict(x=ExpressionScalar('a * b'),
                                        c=ExpressionScalar('a - b'),
                                        d=ExpressionScalar('y')))

        self.assertIs(mock_a, ms._calc_parameter('a'))

        with self.assertRaises(ParameterNotProvidedException):
            ms._calc_parameter('d')

        with mock.patch.object(ms._mapping['x'], 'evaluate_in_scope', return_value=mock_result) as evaluate_in_scope:
            self.assertIs(mock_result, ms._calc_parameter('x'))
            evaluate_in_scope.assert_called_once_with(ds)

        # effective caching tests
        with mock.patch.object(ms._mapping['x'], 'evaluate_in_scope', return_value=mock_result) as evaluate_in_scope:
            self.assertIs(mock_result, ms.get_parameter('x'))
            self.assertIs(mock_result, ms.get_parameter('x'))
            evaluate_in_scope.assert_called_once_with(ds)

    def test_update_constants(self):
        ds = DictScope.from_kwargs(a=1, b=2, c=3, volatile={'c'})
        ds2 = DictScope.from_kwargs(a=1, b=2, c=4, volatile={'c'})
        ms = MappedScope(ds, FrozenDict(x=ExpressionScalar('a * b'),
                                        c=ExpressionScalar('a - b')))
        ms2 = MappedScope(ds2, ms._mapping)

        self.assertIs(ms, ms.change_constants({'f': 1}))

        changes = {'c': 4}
        ms_result = ms.change_constants(changes)
        self.assertEqual(ms2, ms_result)

    def test_volatile_parameters(self):
        ds = DictScope.from_kwargs(a=1, b=2, c=3, d=4, volatile={'c', 'd'})
        ms = MappedScope(ds, FrozenDict(x=ExpressionScalar('a * b'),
                                        c=ExpressionScalar('a - b'),
                                        y=ExpressionScalar('c - a')))
        expected_volatile = FrozenDict(d=ExpressionScalar('d'), y=ExpressionScalar('c - 1'))
        self.assertEqual(expected_volatile, ms.get_volatile_parameters())
        self.assertIs(ms.get_volatile_parameters(), ms.get_volatile_parameters())

    def test_eq(self):
        ds1 = DictScope.from_kwargs(a=1, b=2, c=3, d=4)
        ds2 = DictScope.from_kwargs(a=1, b=2, c=3, d=5)

        mapping1 = FrozenDict(x=ExpressionScalar('a * b'),
                              c=ExpressionScalar('a - b'),
                              y=ExpressionScalar('c - a'))

        mapping2 = FrozenDict(x=ExpressionScalar('a * b'),
                              c=ExpressionScalar('a - b'),
                              y=ExpressionScalar('d - a'))

        self.assertEqual(MappedScope(ds1, mapping1), MappedScope(ds1, mapping1))
        self.assertNotEqual(MappedScope(ds1, mapping1), MappedScope(ds1, mapping2))
        self.assertNotEqual(MappedScope(ds2, mapping1), MappedScope(ds1, mapping1))
        self.assertEqual(MappedScope(ds1, mapping2), MappedScope(ds1, mapping2))
