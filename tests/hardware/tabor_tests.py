import unittest
import numbers
import itertools
from copy import copy, deepcopy
import numpy as np

from qctoolkit.hardware.awgs.tabor import TaborAWGRepresentation, TaborException, TaborProgram, TaborChannelPair,\
    TaborSegment, TaborSequencing, TaborProgramMemory
from qctoolkit.hardware.program import MultiChannelProgram, Loop
from qctoolkit.pulses.instructions import InstructionBlock
from qctoolkit.hardware.util import voltage_to_uint16

from teawg import model_properties_dict

from tests.pulses.sequencing_dummies import DummyWaveform
from tests.hardware import use_dummy_tabor
from tests.hardware.program_tests import LoopTests, WaveformGenerator, MultiChannelTests

hardware_instrument = None
def get_instrument():
    if use_dummy_tabor:
        instrument = TaborAWGRepresentation('dummy_address', reset=True, paranoia_level=2)
        instrument._visa_inst.answers[':OUTP:COUP'] = 'DC'
        instrument._visa_inst.answers[':VOLT'] = '1.0'
        instrument._visa_inst.answers[':FREQ:RAST'] = '1e9'
        instrument._visa_inst.answers[':VOLT:HV'] = '0.7'
        return instrument
    else:
        instrument_address = ('127.0.0.1', )
        if hardware_instrument is None:
            hardware_instrument = TaborAWGRepresentation(instrument_address,
                                            reset=True,
                                            paranoia_level=2)
            hardware_instrument._visa_inst.timeout = 25000

            if not hardware_instrument.is_open:
                raise RuntimeError('Could not connect to instrument')
        return hardware_instrument


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

    def test_depth_0_single_waveform(self):
        program = Loop(waveform=DummyWaveform(defined_channels={'A'}), repetition_count=3)

        t_program = TaborProgram(program, channels=(None, 'A'), markers=(None, None), device_properties=self.instr_props)

        self.assertEqual(t_program.waveform_mode, TaborSequencing.SINGLE)

        self.assertEqual(t_program.get_sequencer_tables(), [[(3, 0, 0)]])
        self.assertEqual(t_program.get_advanced_sequencer_table(), [(1, 1, 0)])

    def test_depth_1_single_waveform(self):
        program = Loop(children=[Loop(waveform=DummyWaveform(defined_channels={'A'}), repetition_count=3)],
                       repetition_count=1)

        t_program = TaborProgram(program, channels=(None, 'A'), markers=(None, None),
                                 device_properties=self.instr_props)

        self.assertEqual(t_program.waveform_mode, TaborSequencing.SINGLE)

        self.assertEqual(t_program.get_sequencer_tables(), [[(3, 0, 0)]])
        self.assertEqual(t_program.get_advanced_sequencer_table(), [(1, 1, 0)])

    def test_depth_1_single_sequence(self):
        program = Loop(children=[Loop(waveform=DummyWaveform(defined_channels={'A'}), repetition_count=3),
                                 Loop(waveform=DummyWaveform(defined_channels={'A'}), repetition_count=4)],
                       repetition_count=1)

        t_program = TaborProgram(program, channels=(None, 'A'), markers=(None, None),
                                 device_properties=self.instr_props)

        self.assertEqual(t_program.waveform_mode, TaborSequencing.SINGLE)

        self.assertEqual(t_program.get_sequencer_tables(), [[(3, 0, 0), (4, 1, 0)]])
        self.assertEqual(t_program.get_advanced_sequencer_table(), [(1, 1, 0)])

    def test_depth_1_single_sequence_2(self):
        """Use the same wf twice"""
        wf_1 = DummyWaveform(defined_channels={'A'})
        wf_2 = DummyWaveform(defined_channels={'A'})

        program = Loop(children=[Loop(waveform=wf_1, repetition_count=3),
                                 Loop(waveform=wf_2, repetition_count=4),
                                 Loop(waveform=wf_1, repetition_count=1)],
                       repetition_count=1)

        t_program = TaborProgram(program, channels=(None, 'A'), markers=(None, None),
                                 device_properties=self.instr_props)

        self.assertEqual(t_program.waveform_mode, TaborSequencing.SINGLE)

        self.assertEqual(t_program.get_sequencer_tables(), [[(3, 0, 0), (4, 1, 0), (1, 0, 0)]])
        self.assertEqual(t_program.get_advanced_sequencer_table(), [(1, 1, 0)])

    def test_depth_1_advanced_sequence_unroll(self):
        wf_1 = DummyWaveform(defined_channels={'A'})
        wf_2 = DummyWaveform(defined_channels={'A'})

        program = Loop(children=[Loop(waveform=wf_1, repetition_count=3),
                                 Loop(waveform=wf_2, repetition_count=4)],
                       repetition_count=5)

        t_program = TaborProgram(program, channels=(None, 'A'), markers=(None, None),
                                 device_properties=self.instr_props)

        self.assertEqual(t_program.waveform_mode, TaborSequencing.ADVANCED)

        # partial unroll of the last segment
        self.assertEqual(t_program.get_sequencer_tables(), [[(3, 0, 0), (3, 1, 0), (1, 1, 0)]])
        self.assertEqual(t_program.get_advanced_sequencer_table(), [(5, 1, 0)])

    def test_depth_1_advanced_sequence(self):
        wf_1 = DummyWaveform(defined_channels={'A'})
        wf_2 = DummyWaveform(defined_channels={'A'})

        program = Loop(children=[Loop(waveform=wf_1, repetition_count=3),
                                 Loop(waveform=wf_2, repetition_count=4),
                                 Loop(waveform=wf_1, repetition_count=1)],
                       repetition_count=5)

        t_program = TaborProgram(program, channels=(None, 'A'), markers=(None, None),
                                 device_properties=self.instr_props)

        self.assertEqual(t_program.waveform_mode, TaborSequencing.ADVANCED)

        # partial unroll of the last segment
        self.assertEqual(t_program.get_sequencer_tables(), [[(3, 0, 0), (4, 1, 0), (1, 0, 0)]])
        self.assertEqual(t_program.get_advanced_sequencer_table(), [(5, 1, 0)])

    def test_advanced_sequence_exceptions(self):
        temp_properties = self.instr_props.copy()
        temp_properties['max_seq_len'] = 5

        program = Loop(children=[Loop(waveform=DummyWaveform(defined_channels={'A'}), repetition_count=1)
                                 for _ in range(temp_properties['max_seq_len']+1)],
                       repetition_count=2)
        with self.assertRaises(TaborException):
            TaborProgram(program.copy_tree_structure(), channels=(None, 'A'), markers=(None, None),
                         device_properties=temp_properties)

        temp_properties['min_seq_len'] = 100
        temp_properties['max_seq_len'] = 120
        with self.assertRaises(TaborException) as exception:
            TaborProgram(program.copy_tree_structure(), channels=(None, 'A'), markers=(None, None),
                         device_properties=temp_properties)
        self.assertEqual(str(exception.exception), 'The algorithm is not smart enough '
                                                   'to make this sequence table longer')

        program = Loop(children=[Loop(children=[Loop(waveform=DummyWaveform(defined_channels={'A'})),
                                                Loop(waveform=DummyWaveform(defined_channels={'A'}))]),
                                 Loop(children=[Loop(waveform=DummyWaveform(defined_channels={'A'})),
                                                Loop(waveform=DummyWaveform(defined_channels={'A'}))])
                                 ])
        with self.assertRaises(TaborException) as exception:
            TaborProgram(program.copy_tree_structure(), channels=(None, 'A'), markers=(None, None),
                         device_properties=temp_properties)
        self.assertEqual(str(exception.exception), 'The algorithm is not smart enough '
                                                   'to make this sequence table longer')

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
        instrument = get_instrument()

        for ch in (1, 2, 3, 4):
            self.assertIsInstance(instrument.sample_rate(ch), numbers.Number)
        with self.assertRaises(TaborException):
            instrument.sample_rate(0)

    def test_amplitude(self):
        with self.assertRaises(TaborException):
            get_instrument().amplitude(6)

        for ch in range(1, 5):
            self.assertIsInstance(get_instrument().amplitude(ch), float)

    def test_select_channel(self):
        with self.assertRaises(TaborException):
            get_instrument().select_channel(6)


@unittest.skipIf(not use_dummy_tabor, "Tests only possible with dummy tabor driver module injection")
class TaborAWGRepresentationDummyBasedTests(unittest.TestCase):
    def test_send_cmd(self):
        inst = get_instrument()

        inst.send_cmd('', paranoia_level=3)
        self.assertEqual(inst.visa_inst.logged_asks[-1], (('*OPC?; :SYST:ERR?',), {}))

        inst.visa_inst.answers['enemene'] = '1;2;3;4'

        inst.paranoia_level = 3
        with self.assertRaises(AssertionError):
            inst.send_cmd('enemene?')

        inst.visa_inst.default_answer = '-451, bla'
        with self.assertRaises(RuntimeError):
            inst.send_cmd('')

    def test_trigger(self):
        inst = get_instrument()
        inst.paranoia_level = 0

        inst.logged_commands = []
        inst.trigger()

        self.assertEqual(inst.logged_commands, [((), dict(cmd_str=':TRIG', paranoia_level=inst.paranoia_level))])




class TaborChannelPairTests(unittest.TestCase):
    def test_copy(self):
        channel_pair = TaborChannelPair(get_instrument(), identifier='asd', channels=(1, 2))
        with self.assertRaises(NotImplementedError):
            copy(channel_pair)
        with self.assertRaises(NotImplementedError):
            deepcopy(channel_pair)

    def test_init(self):
        with self.assertRaises(ValueError):
            TaborChannelPair(get_instrument(), identifier='asd', channels=(1, 3))



