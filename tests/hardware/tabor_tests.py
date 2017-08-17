import unittest
import numbers
import itertools
from copy import copy, deepcopy
import numpy as np


from qctoolkit.hardware.awgs.tabor import TaborAWGRepresentation, TaborException, TaborProgram, TaborChannelPair,\
    TaborSegment
from qctoolkit.hardware.program import MultiChannelProgram
from qctoolkit.pulses.instructions import InstructionBlock
from qctoolkit.hardware.util import voltage_to_uint16
from ..hardware import use_dummy_tabor

from teawg import model_properties_dict

from .program_tests import LoopTests, WaveformGenerator, MultiChannelTests

if not use_dummy_tabor:
    # fix on your machine
    possible_addresses = ('127.0.0.1', )
    for instrument_address in possible_addresses:
        instrument = TaborAWGRepresentation(instrument_address,
                                            reset=True,
                                            paranoia_level=2)
        instrument._visa_inst.timeout = 25000
        if instrument.is_open:
            break
    if not instrument.is_open:
        raise RuntimeError('Could not connect to instrument')
else:
    instrument = TaborAWGRepresentation('dummy_address', reset=False, paranoia_level=2)
    instrument._visa_inst.answers[':OUTP:COUP'] = 'DC'
    instrument._visa_inst.answers[':VOLT'] = '1.0'
    instrument._visa_inst.answers[':FREQ:RAST'] = '1e9'


class TaborSegmentTests(unittest.TestCase):
    def test_init(self):
        with self.assertRaises(TaborException):
            TaborSegment(None, None)
        with self.assertRaises(TaborException):
            TaborSegment(np.zeros(5), np.zeros(4))

        ch_a = np.zeros(5)
        ch_b = np.ones(5)

        ts = TaborSegment(ch_a=ch_a, ch_b=ch_b)
        self.assertIs(ts[0], ch_a)
        self.assertIs(ts[1], ch_b)

    def test_num_points(self):
        self.assertEqual(TaborSegment(np.zeros(5), np.zeros(5)).num_points, 5)


class TaborProgramTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.instr_props = model_properties_dict

    @property
    def waveform_data_generator(self):
        return itertools.cycle([np.linspace(-0.5, 0.5, num=192),
                                                        np.concatenate((np.linspace(-0.5, 0.5, num=96),
                                                                        np.linspace(0.5, -0.5, num=96))),
                                                        -0.5*np.cos(np.linspace(0, 2*np.pi, num=192))])

    @property
    def root_loop(self):
        return LoopTests.get_test_loop(WaveformGenerator(num_channels=2,
                                                         waveform_data_generator=self.waveform_data_generator,
                                                         duration_generator=itertools.repeat(4048e-9)))

    def test_init(self):
        prog = MultiChannelProgram(MultiChannelTests().root_block)
        TaborProgram(prog['A'], self.instr_props, ('A', None), (None, None))

        with self.assertRaises(KeyError):
            TaborProgram(prog['A'], self.instr_props, ('A', 'B'), (None, None))

        with self.assertRaises(TaborException):
            TaborProgram(prog['A'], self.instr_props, ('A', 'B'), (None, None, None))
        with self.assertRaises(TaborException):
            TaborProgram(prog['A'], self.instr_props, ('A', 'B', 'C'), (None, None))


    def test_sampled_segments(self):

        def my_gen(gen):
            alternating_on_off = itertools.cycle((np.ones(192), np.zeros(192)))
            chan_gen = gen
            while True:
                for _ in range(2):
                    yield next(chan_gen)
                yield next(alternating_on_off)
                yield np.zeros(192)

        sample_rate = 10**9
        with self.assertRaises(TaborException):
            root_loop = LoopTests.get_test_loop(WaveformGenerator(
                waveform_data_generator=my_gen(self.waveform_data_generator),
                duration_generator=itertools.repeat(12),
                num_channels=4))

            mcp = MultiChannelProgram(InstructionBlock(), tuple())
            mcp.programs[frozenset(('A', 'B', 'C', 'D'))] = root_loop
            TaborProgram(root_loop, self.instr_props, ('A', 'B'), (None, None)).sampled_segments(8000,
                                                                                           (1., 1.),
                                                                                           (0, 0),
                                                                                           (lambda x: x, lambda x: x))

        root_loop = LoopTests.get_test_loop(WaveformGenerator(
            waveform_data_generator=my_gen(self.waveform_data_generator),
            duration_generator=itertools.repeat(192),
            num_channels=4))

        mcp = MultiChannelProgram(InstructionBlock(), tuple())
        mcp.programs[frozenset(('A', 'B', 'C', 'D'))] = root_loop

        prog = TaborProgram(root_loop, self.instr_props, ('A', 'B'), (None, None))

        sampled, sampled_length = prog.sampled_segments(sample_rate, (1., 1.), (0, 0),
                                                        (lambda x: x, lambda x: x))

        self.assertEqual(len(sampled), 3)

        prog = TaborProgram(root_loop, self.instr_props, ('A', 'B'), ('C', None))
        sampled, sampled_length = prog.sampled_segments(sample_rate, (1., 1.), (0, 0),
                                                        (lambda x: x, lambda x: x))
        self.assertEqual(len(sampled), 6)

        iteroe = my_gen(self.waveform_data_generator)
        for i, sampled_seg in enumerate(sampled):
            data = [next(iteroe) for _ in range(4)]
            data = (voltage_to_uint16(data[0], 1., 0., 14), voltage_to_uint16(data[1], 1., 0., 14), data[2], data[3])
            if i % 2 == 0:
                self.assertTrue(np.all(sampled_seg[0] >> 14 == np.ones(192, dtype=np.uint16)))
            else:
                self.assertTrue(np.all(sampled_seg[0] >> 14 == np.zeros(192, dtype=np.uint16)))
            self.assertTrue(np.all(sampled_seg[0] >> 15 == np.zeros(192, dtype=np.uint16)))
            self.assertTrue(np.all(sampled_seg[1] >> 15 == np.zeros(192, dtype=np.uint16)))

            self.assertTrue(np.all(sampled_seg[0] << 2 == data[0] << 2))
            self.assertTrue(np.all(sampled_seg[1] << 2 == data[1] << 2))


class TaborAWGRepresentationTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_sample_rate(self):
        for ch in (1, 2, 3, 4):
            self.assertIsInstance(instrument.sample_rate(ch), numbers.Number)
        with self.assertRaises(TaborException):
            instrument.sample_rate(0)

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


