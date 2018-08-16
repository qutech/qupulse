import unittest

import pandas as pd
import numpy as np

from qctoolkit._program.transformation import LinearTransformation, Transformation


class TransformationStub(Transformation):
    def __call__(self, *args, **kwargs):
        raise NotImplementedError()

    def get_output_channels(self, input_channels):
        raise NotImplementedError()

    @property
    def compare_key(self):
        return id(self)


class LinearTransformationTests(unittest.TestCase):
    def test_compare_key(self):
        trafo_dict = {'transformed_a': {'a': 1, 'b': -1, 'c': 0}, 'transformed_b': {'a': 1, 'b': 1, 'c': 1}}
        trafo_matrix = pd.DataFrame(trafo_dict).T
        trafo = LinearTransformation(trafo_matrix)

        self.assertEqual(trafo_matrix.to_dict(), trafo.compare_key)

    def test_get_output_channels(self):
        trafo_dict = {'transformed_a': {'a': 1, 'b': -1, 'c': 0}, 'transformed_b': {'a': 1, 'b': 1, 'c': 1}}
        trafo_matrix = pd.DataFrame(trafo_dict).T
        trafo = LinearTransformation(trafo_matrix)

        self.assertEqual(trafo.get_output_channels({'a', 'b', 'c'}), {'transformed_a', 'transformed_b'})
        with self.assertRaisesRegex(KeyError, 'Invalid input channels'):
            trafo.get_output_channels({'a', 'b'})

    def test_call(self):
        trafo_dict = {'transformed_a': {'a': 1., 'b': -1., 'c': 0.}, 'transformed_b': {'a': 1., 'b': 1., 'c': 1.}}
        trafo_matrix = pd.DataFrame(trafo_dict).T
        trafo = LinearTransformation(trafo_matrix)

        data = (np.arange(12.) + 1).reshape((3, 4))
        data = pd.DataFrame(data, index=list('abc'))

        transformed = trafo(np.full(4, np.NaN), data)

        expected = np.empty((2, 4))
        expected[0, :] = data.loc['a'] - data.loc['b']
        expected[1, :] = np.sum(data.values, axis=0)

        expected = pd.DataFrame(expected, index=['transformed_a', 'transformed_b'])

        pd.testing.assert_frame_equal(expected, transformed)

        with self.assertRaisesRegex(KeyError, 'Invalid input channels'):
            trafo(np.full(4, np.NaN), data.loc[['a', 'b']])
