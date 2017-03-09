import unittest
import numpy as np

import qctoolkit.hardware.awgs.base as awg
import qctoolkit.hardware.awgs.tektronix as tek
import qctoolkit.pulses as pls


class DummyAWGTest(unittest.TestCase):

    def setUp(self):
        self.pulse_template = pls.TablePulseTemplate()
        self.pulse_template.add_entry('value', 5)
        self.sequencer = pls.Sequencer()
        for i in range(1,12):
            pars = dict(value=i)
            self.sequencer.push(self.pulse_template, pars, channel_mapping=dict(default='default'))
        self.program = self.sequencer.build()

    def test_ProgramOverwriteException(self):
        dummy = awg.DummyAWG(100)
        dummy.upload('program', self.program, [], [], [])
        with self.assertRaises(awg.ProgramOverwriteException):
            dummy.upload('program', self.program, [], [], [])


class TektronixAWGTest(unittest.TestCase):

    def setUp(self):
        self.pulse_template = pls.TablePulseTemplate()
        self.pulse_template.add_entry('value', 5)
        self.sequencer = pls.Sequencer()
        for i in range(1,12):
            pars = dict(value=i)
            self.sequencer.push(self.pulse_template, pars, channel_mapping=dict(default='default'))
        self.program = self.sequencer.build()

    @unittest.skip
    def test_ProgramOverwriteException(self):
        dummy = tek.TektronixAWG('127.0.0.1', 8000, 100000, simulation=True)
        dummy.upload('program', self.program)
        with self.assertRaises(awg.ProgramOverwriteException):
            dummy.upload('program', self.program)

    @unittest.skip
    def test_upload(self):
        dummy = tek.TektronixAWG('127.0.0.1', 8000, 100000, simulation=True)
        dummy.upload('program', self.program)
        programs = dummy.programs
        self.assertEqual(programs, ['program'])
