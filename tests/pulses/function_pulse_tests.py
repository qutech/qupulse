import unittest

from qctoolkit.pulses.function_pulse_template import FunctionPulseTemplate

from tests.serialization_dummies import DummySerializer
from tests.pulses.sequencing_dummies import DummySequencer, DummyInstructionBlock, DummyInterpolationStrategy, DummyParameter

class FunctionPulseTest(unittest.TestCase):
    def setUp(self):
        self.s = 'a + b'
        self.s2 = 'c'
        self.pars = dict(a=1, b=2, c=3)
        self.fpt = FunctionPulseTemplate(self.s, self.s2)

    def test_get_pulse_length(self):
        self.assertEqual(self.fpt.get_pulse_length(self.pars), 3)

    def test_get_measurement_windows(self):
        self.assertEqual(self.fpt.get_measurement_windows(self.pars), None)

        fpt2 = FunctionPulseTemplate(self.s, self.s2, measurement=True)
        self.assertEqual(fpt2.get_measurement_windows(self.pars), [(0, 3)])

    def test_serialization_data(self):
        expected_data = dict(type='FunctionPulseTemplate',
                             parameter_names=set(['a', 'b', 'c']),
                             duration_expression=self.s2,
                             expression=self.s,
                             measurement=False)
        self.assertEqual(expected_data, self.fpt.get_serialization_data(DummySerializer()))

class FunctionPulseSequencingTest(unittest.TestCase):
    def setUp(self):
        unittest.TestCase.setUp(self)
        self.f = "a + t"
        self.duration = "y"
        self.args = dict(a=1,y=3)
        self.fpt = FunctionPulseTemplate(self.f, self.duration)
    
    def test_build_sequence(self):
        ib = DummyInstructionBlock()
        self.fpt.build_sequence(DummySequencer(), self.args, ib)
        self.assertTrue(self.args["y"],len(ib))
        