import unittest
import numpy as np

import qctoolkit.hardware.awgs.awg as awg
import qctoolkit.pulses as pls


class DummyAWGTest(unittest.TestCase):

    def setUp(self):
        self.pulse_template = pls.TablePulseTemplate()
        self.pulse_template.add_entry('value', 5)
        self.sequencer = pls.Sequencer()
        for i in range(1,12):
            pars = dict(value=i)
            self.sequencer.push(self.pulse_template, pars)
        self.program = self.sequencer.build()

    def test_OutOfMemoryException(self):
        dummy = awg.DummyAWG(10)
        with self.assertRaises(awg.OutOfWaveformMemoryException):
            dummy.upload('program', self.program)

    def test_ProgramOverwriteException(self):
        dummy = awg.DummyAWG(100)
        dummy.upload('program', self.program)
        with self.assertRaises(awg.ProgramOverwriteException):
            dummy.upload('program', self.program)

    def test_upload(self):
        dummy = awg.DummyAWG(100)
        dummy.upload('program',self.program)
        memory_part = [None for i in range(89)]
        self.assertEqual(dummy._DummyAWG__waveform_memory[11:], memory_part)
        self.assertEqual(dummy.programs, set(['program']))
