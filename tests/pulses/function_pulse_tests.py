import unittest

import numpy as np

from qctoolkit.expressions import Expression
from qctoolkit.pulses.function_pulse_template import FunctionPulseTemplate,\
    FunctionWaveform
from qctoolkit.pulses.instructions import InstructionBlock
from qctoolkit.pulses.measurements import Measurement
from qctoolkit.pulses.sequencing import Sequencer
from tests.serialization_dummies import DummySerializer

class FunctionPulseTest(unittest.TestCase):
    def setUp(self):
        self.s = 'a + b'
        self.s2 = 'c'
        self.pars = dict(a=1, b=2, c=3)
        self.fpt = FunctionPulseTemplate(self.s, self.s2)

    def test_get_pulse_length(self):
        self.assertEqual(self.fpt.get_pulse_length(self.pars), 3)

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
        self.f = "a * t"
        self.duration = "y"
        self.args = dict(a=3,y=1)
        self.fpt = FunctionPulseTemplate(self.f, self.duration)
    
    def test_build_sequence(self):
        ib = InstructionBlock()
        seq = Sequencer()
        cond = None
        self.fpt.build_sequence(seq, self.args, cond, ib)
        
class FunctionWaveformTest(unittest.TestCase):
    def test_sample(self):
        f = Expression("(t+1)**b")
        length = Expression("c**b")
        par = {"b":2,"c":10}
        fw = FunctionWaveform(par, f, length, Measurement(FunctionPulseTemplate("(t+1)**b", "c**b")))
        a = np.arange(4)
        self.assertEqual(list(fw.sample(a)), [1,4,9,16])
        