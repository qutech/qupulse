import unittest
from unittest import mock

import pandas as pd
import numpy as np

from qupulse._program.transformation import LinearTransformation, Transformation, IdentityTransformation,\
    ChainedTransformation, chain_transformations


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

        with self.assertRaises(AssertionError):
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


class IdentityTransformationTests(unittest.TestCase):
    def test_compare_key(self):
        self.assertIsNone(IdentityTransformation().compare_key)

    def test_singleton(self):
        self.assertIs(IdentityTransformation(), IdentityTransformation())

    def test_call(self):
        time = np.arange(12)
        data = (np.arange(12.) + 1).reshape((3, 4))
        data = pd.DataFrame(data, index=list('abc'))

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

        data_in = pd.DataFrame(data, index=list('abc'))
        data_0 = data_in + 42
        data_1 = data_0 + 42
        data_2 = data_1 + 42
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
