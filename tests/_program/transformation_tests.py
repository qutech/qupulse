import unittest
from unittest import mock

import numpy as np

from qupulse._program.transformation import LinearTransformation, Transformation, IdentityTransformation,\
    ChainedTransformation, ParallelConstantChannelTransformation, chain_transformations, OffsetTransformation,\
    ScalingTransformation


class TransformationStub(Transformation):
    def __call__(self, *args, **kwargs):
        raise NotImplementedError()

    def get_output_channels(self, input_channels):
        raise NotImplementedError()

    def get_input_channels(self, output_channels):
        raise NotImplementedError()

    @property
    def compare_key(self):
        return id(self)


class TransformationTests(unittest.TestCase):
    def test_chain(self):
        trafo = TransformationStub()

        self.assertIs(trafo.chain(IdentityTransformation()), trafo)

        with mock.patch('qupulse._program.transformation.chain_transformations',
                        return_value='asd') as chain_transformations:
            self.assertEqual(trafo.chain(trafo), 'asd')
            chain_transformations.assert_called_once_with(trafo, trafo)


class LinearTransformationTests(unittest.TestCase):
    def test_compare_key_and_init(self):
        in_chs = ('a', 'b', 'c')
        out_chs = ('transformed_a', 'transformed_b')
        matrix = np.array([[1, -1, 0], [1, 1, 1]])

        with self.assertRaisesRegex(ValueError, 'Shape'):
            LinearTransformation(matrix, in_chs[:-1], out_chs)
        trafo = LinearTransformation(matrix, in_chs, out_chs)

        in_chs_2 = ('a', 'c', 'b')
        out_chs_2 = ('transformed_b', 'transformed_a')
        matrix_2 = np.array([[1, 1, 1], [1, 0, -1]])
        trafo_2 = LinearTransformation(matrix_2, in_chs_2, out_chs_2)

        self.assertEqual(trafo.compare_key, trafo_2.compare_key)
        self.assertEqual(trafo, trafo_2)
        self.assertEqual(hash(trafo), hash(trafo_2))
        self.assertEqual(trafo.compare_key, (in_chs, out_chs, matrix.tobytes()))

    def test_from_pandas(self):
        try:
            import pandas as pd
        except ImportError:
            raise unittest.SkipTest('pandas package not present')

        trafo_dict = {'transformed_a': {'a': 1, 'b': -1, 'c': 0}, 'transformed_b': {'a': 1, 'b': 1, 'c': 1}}
        trafo_df = pd.DataFrame(trafo_dict).T
        trafo = LinearTransformation.from_pandas(trafo_df)

        trafo_matrix = np.array([[1, -1, 0], [1, 1, 1]])

        self.assertEqual(trafo._input_channels, tuple('abc'))
        self.assertEqual(trafo._output_channels, ('transformed_a', 'transformed_b'))
        np.testing.assert_equal(trafo_matrix, trafo._matrix)

    def test_get_output_channels(self):
        in_chs = ('a', 'b', 'c')
        out_chs = ('transformed_a', 'transformed_b')
        matrix = np.array([[1, -1, 0], [1, 1, 1]])
        trafo = LinearTransformation(matrix, in_chs, out_chs)

        self.assertEqual(trafo.get_output_channels({'a', 'b', 'c'}), {'transformed_a', 'transformed_b'})
        with self.assertRaisesRegex(KeyError, 'Invalid input channels'):
            trafo.get_output_channels({'a', 'b'})

    def test_get_input_channels(self):
        in_chs = ('a', 'b', 'c')
        out_chs = ('transformed_a', 'transformed_b')
        matrix = np.array([[1, -1, 0], [1, 1, 1]])
        trafo = LinearTransformation(matrix, in_chs, out_chs)

        self.assertEqual(trafo.get_input_channels({'transformed_a'}), {'a', 'b', 'c'})
        self.assertEqual(trafo.get_input_channels({'transformed_a', 'd'}), {'a', 'b', 'c', 'd'})
        self.assertEqual(trafo.get_input_channels({'d'}), {'d'})
        with self.assertRaisesRegex(KeyError, 'Is input channel'):
            self.assertEqual(trafo.get_input_channels({'transformed_a', 'a'}), {'a', 'b', 'c', 'd'})

        in_chs = ('a', 'b', 'c')
        out_chs = ('a', 'b', 'c')
        matrix = np.eye(3)

        trafo = LinearTransformation(matrix, in_chs, out_chs)
        in_set = {'transformed_a'}
        self.assertIs(trafo.get_input_channels(in_set), in_set)
        self.assertEqual(trafo.get_input_channels({'transformed_a', 'a'}), {'transformed_a', 'a', 'b', 'c'})

    def test_call(self):
        in_chs = ('a', 'b', 'c')
        out_chs = ('transformed_a', 'transformed_b')
        matrix = np.array([[1, -1, 0], [1, 1, 1]])
        trafo = LinearTransformation(matrix, in_chs, out_chs)

        raw_data = (np.arange(12.) + 1).reshape((3, 4))
        data = dict(zip('abc', raw_data))

        data['ignored'] = np.arange(116., 120.)

        transformed = trafo(np.full(4, np.NaN), data)

        expected = {'transformed_a': data['a'] - data['b'],
                    'transformed_b': np.sum(raw_data, axis=0),
                    'ignored': np.arange(116., 120.)}

        np.testing.assert_equal(expected, transformed)

        data.pop('c')
        with self.assertRaisesRegex(KeyError, 'Invalid input channels'):

            trafo(np.full(4, np.NaN), data)

        in_chs = ('a', 'b', 'c')
        out_chs = ('a', 'b', 'c')
        matrix = np.eye(3)
        trafo = LinearTransformation(matrix, in_chs, out_chs)

        data_in = {'ignored': np.arange(116., 120.)}
        transformed = trafo(np.full(4, np.NaN), data_in)
        np.testing.assert_equal(transformed, data_in)
        self.assertIs(data_in['ignored'], transformed['ignored'])

    def test_repr(self):
        in_chs = ('a', 'b', 'c')
        out_chs = ('transformed_a', 'transformed_b')
        matrix = np.array([[1, -1, 0], [1, 1, 1]])
        trafo = LinearTransformation(matrix, in_chs, out_chs)
        self.assertEqual(trafo, eval(repr(trafo)))


class IdentityTransformationTests(unittest.TestCase):
    def test_compare_key(self):
        self.assertIsNone(IdentityTransformation().compare_key)

    def test_singleton(self):
        self.assertIs(IdentityTransformation(), IdentityTransformation())

    def test_call(self):
        time = np.arange(12)
        data = dict(zip('abc',(np.arange(12.) + 1).reshape((3, 4))))
        self.assertIs(IdentityTransformation()(time, data), data)

    def test_output_channels(self):
        chans = {'a', 'b'}
        self.assertIs(IdentityTransformation().get_output_channels(chans), chans)

    def test_input_channels(self):
        chans = {'a', 'b'}
        self.assertIs(IdentityTransformation().get_input_channels(chans), chans)

    def test_chain(self):
        trafo = TransformationStub()
        self.assertIs(IdentityTransformation().chain(trafo), trafo)

    def test_repr(self):
        trafo = IdentityTransformation()
        self.assertEqual(trafo, eval(repr(trafo)))


class ChainedTransformationTests(unittest.TestCase):
    def test_init_and_properties(self):
        trafos = TransformationStub(), TransformationStub(), TransformationStub()
        chained = ChainedTransformation(*trafos)

        self.assertEqual(chained.transformations, trafos)
        self.assertIs(chained.transformations, chained.compare_key)

    def test_get_output_channels(self):
        trafos = TransformationStub(), TransformationStub(), TransformationStub()
        chained = ChainedTransformation(*trafos)
        chans = {1}, {2}, {3}

        with mock.patch.object(trafos[0], 'get_output_channels', return_value=chans[0]) as get_output_channels_0,\
                mock.patch.object(trafos[1], 'get_output_channels', return_value=chans[1]) as get_output_channels_1,\
                mock.patch.object(trafos[2], 'get_output_channels', return_value=chans[2]) as get_output_channels_2:
            outs = chained.get_output_channels({0})

            self.assertIs(outs, chans[2])
            get_output_channels_0.assert_called_once_with({0})
            get_output_channels_1.assert_called_once_with({1})
            get_output_channels_2.assert_called_once_with({2})

    def test_get_input_channels(self):
        trafos = TransformationStub(), TransformationStub(), TransformationStub()
        chained = ChainedTransformation(*trafos)
        chans = {1}, {2}, {3}

        # note reverse trafos order
        with mock.patch.object(trafos[2], 'get_input_channels', return_value=chans[0]) as get_input_channels_0,\
                mock.patch.object(trafos[1], 'get_input_channels', return_value=chans[1]) as get_input_channels_1,\
                mock.patch.object(trafos[0], 'get_input_channels', return_value=chans[2]) as get_input_channels_2:
            outs = chained.get_input_channels({0})

            self.assertIs(outs, chans[2])
            get_input_channels_0.assert_called_once_with({0})
            get_input_channels_1.assert_called_once_with({1})
            get_input_channels_2.assert_called_once_with({2})

    def test_call(self):
        trafos = TransformationStub(), TransformationStub(), TransformationStub()
        chained = ChainedTransformation(*trafos)

        time = np.arange(12)
        data = (np.arange(12.) + 1).reshape((3, 4))

        data_in = dict(zip('abc', data))
        data_0 = dict(zip('abc', data + 42))
        data_1 = dict(zip('abc', data + 2*42))
        data_2 = dict(zip('abc', data + 3*42))
        with mock.patch('tests._program.transformation_tests.TransformationStub.__call__',
                        side_effect=[data_0, data_1, data_2]) as call:
            outs = chained(time, data_in)

            self.assertIs(outs, data_2)
            self.assertEqual(call.call_count, 3)
            for ((time_arg, data_arg), kwargs), expected_data in zip(call.call_args_list,
                                                                     [data_in, data_0, data_1]):
                self.assertEqual(kwargs, {})
                self.assertIs(time, time_arg)
                self.assertIs(expected_data, data_arg)

    def test_chain(self):
        trafos = TransformationStub(), TransformationStub()
        trafo = TransformationStub()
        chained = ChainedTransformation(*trafos)

        with mock.patch('qupulse._program.transformation.chain_transformations',
                        return_value='asd') as chain_transformations:
            self.assertEqual(chained.chain(trafo), 'asd')
            chain_transformations.assert_called_once_with(*trafos, trafo)

    def test_repr(self):
        trafo = ChainedTransformation(ScalingTransformation({'a': 1.1}), OffsetTransformation({'b': 6.6}))
        self.assertEqual(trafo, eval(repr(trafo)))


class ParallelConstantChannelTransformationTests(unittest.TestCase):
    def test_init(self):
        channels = {'X': 2, 'Y': 4.4}

        trafo = ParallelConstantChannelTransformation(channels)

        self.assertEqual(trafo._channels, channels)
        self.assertTrue(all(isinstance(v, float) for v in trafo._channels.values()))

        self.assertEqual(trafo.compare_key, (('X', 2.), ('Y', 4.4)))

        self.assertEqual(trafo.get_input_channels(set()), set())
        self.assertEqual(trafo.get_input_channels({'X'}), set())
        self.assertEqual(trafo.get_input_channels({'Z'}), {'Z'})
        self.assertEqual(trafo.get_input_channels({'X', 'Z'}), {'Z'})

        self.assertEqual(trafo.get_output_channels(set()), {'X', 'Y'})
        self.assertEqual(trafo.get_output_channels({'X'}), {'X', 'Y'})
        self.assertEqual(trafo.get_output_channels({'X', 'Z'}), {'X', 'Y', 'Z'})

    def test_trafo(self):
        channels = {'X': 2, 'Y': 4.4}
        trafo = ParallelConstantChannelTransformation(channels)

        n_points = 17
        time = np.arange(17, dtype=float)

        expected_overwrites = {'X': np.full((n_points,), 2.),
                               'Y': np.full((n_points,), 4.4)}

        empty_input_result = trafo(time, {})
        np.testing.assert_equal(empty_input_result, expected_overwrites)

        z_input_result = trafo(time, {'Z': np.sin(time)})
        np.testing.assert_equal(z_input_result, {'Z': np.sin(time), **expected_overwrites})

        x_input_result = trafo(time, {'X': np.cos(time)})
        np.testing.assert_equal(empty_input_result, expected_overwrites)

        x_z_input_result = trafo(time, {'X': np.cos(time), 'Z': np.sin(time)})
        np.testing.assert_equal(z_input_result, {'Z': np.sin(time), **expected_overwrites})

    def test_repr(self):
        channels = {'X': 2, 'Y': 4.4}
        trafo = ParallelConstantChannelTransformation(channels)
        self.assertEqual(trafo, eval(repr(trafo)))


class TestChaining(unittest.TestCase):
    def test_identity_result(self):
        self.assertIs(chain_transformations(), IdentityTransformation())

        self.assertIs(chain_transformations(IdentityTransformation(), IdentityTransformation()),
                      IdentityTransformation())

    def test_single_transformation(self):
        trafo = TransformationStub()

        self.assertIs(chain_transformations(trafo), trafo)
        self.assertIs(chain_transformations(trafo, IdentityTransformation()), trafo)

    def test_denesting(self):
        trafo = TransformationStub()
        chained = ChainedTransformation(TransformationStub(), TransformationStub())

        expected = ChainedTransformation(trafo, *chained.transformations, trafo)
        result = chain_transformations(trafo, chained, trafo)

        self.assertEqual(expected, result)

    def test_chaining(self):
        trafo = TransformationStub()

        expected = ChainedTransformation(trafo, trafo)

        result = chain_transformations(trafo, IdentityTransformation(), trafo)

        self.assertEqual(result, expected)


class TestOffsetTransformation(unittest.TestCase):
    def setUp(self) -> None:
        self.offsets = {'A': 1., 'B': 1.2}

    def test_init(self):
        trafo = OffsetTransformation(self.offsets)
        # test copy
        self.assertIsNot(trafo._offsets, self.offsets)

        self.assertEqual(trafo._offsets, self.offsets)

    def test_get_input_channels(self):
        trafo = OffsetTransformation(self.offsets)
        channels = {'A', 'C'}
        self.assertIs(channels, trafo.get_input_channels(channels))
        self.assertIs(channels, trafo.get_output_channels(channels))

    def test_compare_key(self):
        trafo = OffsetTransformation(self.offsets)
        _ = hash(trafo)
        self.assertEqual(frozenset([('A', 1.), ('B', 1.2)]), trafo.compare_key)

    def test_trafo(self):
        trafo = OffsetTransformation(self.offsets)

        time = np.asarray([.5, .6])
        in_data = {'A': np.asarray([.1, .2]), 'C': np.asarray([3., 4.])}

        expected = {'A': np.asarray([1.1, 1.2]), 'C': in_data['C']}

        out_data = trafo(time, in_data)

        self.assertIs(expected['C'], out_data['C'])
        np.testing.assert_equal(expected, out_data)

    def test_repr(self):
        trafo = OffsetTransformation(self.offsets)
        self.assertEqual(trafo, eval(repr(trafo)))


class TestScalingTransformation(unittest.TestCase):
    def setUp(self) -> None:
        self.scales = {'A': 1.5, 'B': 1.2}

    def test_init(self):
        trafo = ScalingTransformation(self.scales)
        # test copy
        self.assertIsNot(trafo._factors, self.scales)
        self.assertEqual(trafo._factors, self.scales)

    def test_get_input_channels(self):
        trafo = ScalingTransformation(self.scales)
        channels = {'A', 'C'}
        self.assertIs(channels, trafo.get_input_channels(channels))
        self.assertIs(channels, trafo.get_output_channels(channels))

    def test_compare_key(self):
        trafo = OffsetTransformation(self.scales)
        _ = hash(trafo)
        self.assertEqual(frozenset([('A', 1.5), ('B', 1.2)]), trafo.compare_key)

    def test_trafo(self):
        trafo = ScalingTransformation(self.scales)

        time = np.asarray([.5, .6])
        in_data = {'A': np.asarray([.1, .2]), 'C': np.asarray([3., 4.])}
        expected = {'A': np.asarray([.1 * 1.5, .2 * 1.5]), 'C': in_data['C']}

        out_data = trafo(time, in_data)

        self.assertIs(expected['C'], out_data['C'])
        np.testing.assert_equal(expected, out_data)

    def test_repr(self):
        trafo = OffsetTransformation(self.scales)
        self.assertEqual(trafo, eval(repr(trafo)))
