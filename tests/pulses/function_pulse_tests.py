import unittest

from qctoolkit.pulses.function_pulse_template import FunctionPulseTemplate,\
    FunctionWaveform
from qctoolkit.pulses.sequencing import Sequencer
from qctoolkit.pulses.instructions import InstructionBlock
from qctoolkit.pulses.parameters import ConstantParameter

from tests.serialization_dummies import DummySerializer
from qctoolkit.expressions import Expression

import numpy as np

class FunctionPulseTest(unittest.TestCase):
    def setUp(self):
        self.s = 'a + b'
        self.s2 = 'c'
        self.pars = dict(a=ConstantParameter(1), b=ConstantParameter(2), c=ConstantParameter(3))
        self.fpt = FunctionPulseTemplate(self.s, self.s2)

    def test_get_pulse_length(self):
        self.assertEqual(self.fpt.get_pulse_length(self.pars), 3)

#    def test_get_measurement_windows(self):
#        self.assertEqual(self.fpt.get_measurement_windows(self.pars), None)
#
#        fpt2 = FunctionPulseTemplate(self.s, self.s2, measurement=True)
#        self.assertEqual(fpt2.get_measurement_windows(self.pars), [(0, 3)])

    def test_serialization_data(self):
        expected_data = dict(type='FunctionPulseTemplate',
                             parameter_names=set(['a', 'b', 'c']),
                             duration_expression=str(self.s2),
                             expression=str(self.s),
                             measurement=False)
        self.assertEqual(expected_data, self.fpt.get_serialization_data(DummySerializer(serialize_callback=lambda x: str(x))))


class FunctionPulseSequencingTest(unittest.TestCase):
    def setUp(self):
        unittest.TestCase.setUp(self)
        self.f = "a * t"
        self.duration = "y"
        self.args = dict(a=ConstantParameter(3),y=ConstantParameter(1))
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
        fw = FunctionWaveform(par,f,length)
        a = np.arange(4)
        self.assertEqual(list(fw.sample(a)), [1,4,9,16])
        