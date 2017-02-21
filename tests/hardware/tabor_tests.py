import unittest
from qctoolkit.hardware.awgs.tabor import TaborAWGRepresentation, TaborException, TaborProgram, TaborChannelPair
from qctoolkit.hardware.program import MultiChannelProgram
import numbers
import itertools
import numpy as np
from copy import copy, deepcopy
from teawg import model_properties_dict
from qctoolkit.hardware.util import voltage_to_uint16

from .program_tests import LoopTests


try:
    instrument_address = '127.0.0.1'
    #instrument_address = '192.168.1.223'
    #instrument_address = 'ASRL10::INSTR'
    instrument = TaborAWGRepresentation(instrument_address)
    instrument._visa_inst.timeout = 25000
except:
    instrument = None


class TaborProgramTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.instr_props = next(model_properties_dict.values().__iter__())

    @property
    def waveform_data_generator(self):
        return itertools.cycle([np.linspace(-0.5, 0.5, num=4048),
                                                        np.concatenate((np.linspace(-0.5, 0.5, num=2024),
                                                                        np.linspace(0.5, -0.5, num=2024))),
                                                        -0.5*np.cos(np.linspace(0, 2*np.pi, num=4048))])

    @property
    def root_block(self):
        return LoopTests(waveform_data_generator=self.waveform_data_generator, waveform_duration=4048e-9).root_block

    @property
    def working_root_block(self):
        block = self.root_block

    def test_init(self):
        prog = MultiChannelProgram(self.root_block)
        TaborProgram(prog, self.instr_props, ('A', 'B'), (None, None))
        with self.assertRaises(Exception):
            TaborProgram(prog, self.instr_props, ('A',), (None, None))

    @unittest.skip
    def test_setup_single_waveform_mode(self):
        pass

    def test_sampled_segments(self):

        def my_gen(gen):
            alternating_on_off = itertools.cycle((np.ones(4048), np.zeros(4048)))
            chan_gen = gen
            while True:
                for _ in range(2):
                    yield next(chan_gen)
                yield next(alternating_on_off)
                yield np.zeros(4048)

        sample_rate = 8096
        with self.assertRaises(TaborException):
            TaborProgram(MultiChannelProgram(
                LoopTests(waveform_data_generator=my_gen(self.waveform_data_generator),
                          waveform_duration=1e-9,
                          num_channels=4).root_block
            ), self.instr_props, ('A', 'B'), (None, None)).sampled_segments(8000, (1., 1.), (0, 0))

        mcp = MultiChannelProgram(
            LoopTests(waveform_data_generator=my_gen(self.waveform_data_generator),
                      waveform_duration=0.5,
                      num_channels=4).root_block
        )
        prog = TaborProgram(mcp, self.instr_props, ('A', 'B'), (None, None))

        sampled, sampled_length = prog.sampled_segments(sample_rate, (1., 1.), (0, 0))

        self.assertEqual(len(sampled), 3)

        prog = TaborProgram(mcp, self.instr_props, ('A', 'B'), ('C', None))
        sampled, sampled_length = prog.sampled_segments(sample_rate, (1., 1.), (0, 0))
        self.assertEqual(len(sampled), 6)

        iteroe = my_gen(self.waveform_data_generator)
        for i, sampled_seg in enumerate(sampled):
            data = [next(iteroe) for _ in range(4)]
            data = (voltage_to_uint16(data[0], 1., 0., 14), voltage_to_uint16(data[1], 1., 0., 14), data[2], data[3])
            if i % 2 == 0:
                self.assertTrue(np.all(sampled_seg[1] >> 14 == np.ones(4048, dtype=np.uint16)))
            else:
                self.assertTrue(np.all(sampled_seg[1] >> 14 == np.zeros(4048, dtype=np.uint16)))
            self.assertTrue(np.all(sampled_seg[0] >> 15 == np.zeros(4048, dtype=np.uint16)))

            self.assertTrue(np.all(sampled_seg[0] << 2 == data[0] << 2))
            self.assertTrue(np.all(sampled_seg[1] << 2 == data[1] << 2))

    @unittest.skipIf(instrument is None, "Instrument not present")
    def test_upload(self):
        prog = MultiChannelProgram(self.root_block)
        program = TaborProgram(prog, instrument.dev_properties, ('A', 'B'))

        program.upload_to_device(instrument, (1, 2))


class TaborAWGRepresentationTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @unittest.skipIf(instrument is None, "Instrument not present")
    def test_sample_rate(self):
        if not instrument.is_open:
            self.skipTest("No instrument found.")
        for ch in (1, 2, 3, 4):
            self.assertIsInstance(instrument.sample_rate(ch), numbers.Number)
        with self.assertRaises(TaborException):
            instrument.sample_rate(0)

    @unittest.skipIf(instrument is None, "Instrument not present")
    def test_amplitude(self):
        for ch in range(1, 5):
            self.assertIsInstance(instrument.amplitude(ch), float)


class TaborChannelPairTests(unittest.TestCase):
    def test_copy(self):
        channel_pair = TaborChannelPair(instrument, identifier='asd', channels=(1, 2))
        with self.assertRaises(NotImplementedError):
            copy(channel_pair)
        with self.assertRaises(NotImplementedError):
            deepcopy(channel_pair)


