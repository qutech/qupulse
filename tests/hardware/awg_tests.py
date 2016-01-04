import unittest
import numpy as np

import qctoolkit.hardware.awg as awg
import qctoolkit.pulses as pls

class DummyAWGTest(unittest.TestCase):
    def setUp(self):
        self.awg = awg.DummyAWG(10)
        self.pulse_template = pls.TablePulseTemplate()
        self.pulse_template.add_entry('value', 5)
        self.sequencer = pls.Sequencer()
        for i in range(1,12):
            pars = dict(value=i)
            sequencer.push(self.pulse_template, pars)
        self.program = self.sequencer.build()

    def test_outofmemoryexception(self):
        with self.assertRaises(awg.OutOfMemoryException):
            self.awg.upload('program', self.program)
