import unittest
import numpy as np

from qctoolkit.experiment.Experiment import Experiment, expand_parameter_space
from qctoolkit.pulses.table_pulse_template import TablePulseTemplate

class ExperimentTest(unittest.TestCase):
    def setUp(self):
        self.axis_a = ('a', np.arange(2))
        self.axis_b = ('b', np.arange(3))
        self.axis_c = ('c', np.arange(2))

    def test_parameter_space_expansion(self):
        expected_tuples = np.array([[0,0,0],
                             [0,0,1],
                             [0,1,0],
                             [0,1,1],
                             [0,2,0],
                             [0,2,1],
                             [1,0,0],
                             [1,0,1],
                             [1,1,0],
                             [1,1,1],
                             [1,2,0],
                             [1,2,1]])
        expected_vectors = [np.arange(2),np.arange(3),np.arange(2)]
        names, vectors, tuples = expand_parameter_space([self.axis_a,
                                         self.axis_b,
                                         self.axis_c])
        self.assertEqual(names, ['a', 'b', 'c'])
        for v, ev in zip(vectors, expected_vectors):
            self.assertTrue(np.all(v == ev))
        self.assertTrue(np.all(tuples == expected_tuples))
