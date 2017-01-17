import unittest
from qctoolkit.hardware.awgs.tabor import TaborAWG, TaborException, TaborProgram
from qctoolkit.hardware.program import MultiChannelProgram
import numbers
import itertools
import numpy as np

from .program_tests import LoopTests

instrument_address = '127.0.0.1'
#instrument_address = '192.168.1.223'
instrument = TaborAWG(instrument_address)
instrument._visa_inst.timeout = 25000

class TaborAWGTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)



    def test_samplerate(self):
        if not instrument.is_open:
            self.skipTest("No instrument found.")
        for ch in (1, 2, 3, 4):
            self.assertIsInstance(instrument.samplerate(ch), numbers.Number)
        with self.assertRaises(TaborException):
            instrument.samplerate(0)

    def test_amplitude(self):
        for ch in range(1, 5):
            self.assertIsInstance(instrument.amplitude(ch), float)


class TaborProgramTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.waveform_data_generator = itertools.cycle([np.linspace(-0.5, 0.5, num=4048),
                                                        np.concatenate((np.linspace(-0.5, 0.5, num=2024),
                                                                        np.linspace(0.5, -0.5, num=2024))),
                                                        -0.5*np.cos(np.linspace(0, 2*np.pi, num=4048))])

    @property
    def root_block(self):
        return LoopTests(waveform_data_generator=self.waveform_data_generator, waveform_duration=4048e-9).root_block

    @property
    def working_root_block(self):
        block = self.root_block

    def setUp(self):
        if not instrument.is_open:
            self.skipTest("Instrument not found.")

    def test_init(self):
        prog = MultiChannelProgram(self.root_block)
        TaborProgram(prog, instrument.dev_properties, {'A', 'B'})

    def test_upload(self):
        prog = MultiChannelProgram(self.root_block)
        program = TaborProgram(prog, instrument.dev_properties, ('A', 'B'))

        program.upload_to_device(instrument, (1, 2))






