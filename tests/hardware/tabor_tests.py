import unittest
import itertools
import numpy as np

from teawg import model_properties_dict

from qctoolkit.hardware.awgs.tabor import TaborException, TaborProgram, \
    TaborSegment, TaborSequencing, with_configuration_guard
from qctoolkit.hardware.program import MultiChannelProgram, Loop
from qctoolkit.pulses.instructions import InstructionBlock
from qctoolkit.hardware.util import voltage_to_uint16

from tests.pulses.sequencing_dummies import DummyWaveform
from tests.hardware.program_tests import LoopTests, WaveformGenerator, MultiChannelTests


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

        self.instr_props = model_properties_dict['WX2184C']

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

    def test_markers(self):
        self.assertEqual(TaborProgram(self.root_loop, self.instr_props, ('A', None), (None, 'B')).markers, (None, 'B'))

    def test_channels(self):
        self.assertEqual(TaborProgram(self.root_loop, self.instr_props, ('A', None), (None, 'B')).channels, ('A', None))

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


class ConfigurationGuardTest(unittest.TestCase):
    class DummyChannelPair:
        def __init__(self, test_obj: unittest.TestCase):
            self.test_obj = test_obj
            self._configuration_guard_count = 0
            self.is_in_config_mode = False

        def _enter_config_mode(self):
            self.test_obj.assertFalse(self.is_in_config_mode)
            self.test_obj.assertEqual(self._configuration_guard_count, 0)
            self.is_in_config_mode = True

        def _exit_config_mode(self):
            self.test_obj.assertTrue(self.is_in_config_mode)
            self.test_obj.assertEqual(self._configuration_guard_count, 0)
            self.is_in_config_mode = False

        @with_configuration_guard
        def guarded_method(self, counter=5, throw=False):
            self.test_obj.assertTrue(self.is_in_config_mode)
            if counter > 0:
                return self.guarded_method(counter - 1, throw) + 1
            if throw:
                raise RuntimeError()
            return 0

    def test_config_guard(self):
        channel_pair = ConfigurationGuardTest.DummyChannelPair(self)

        for i in range(5):
            self.assertEqual(channel_pair.guarded_method(i), i)

        with self.assertRaises(RuntimeError):
            channel_pair.guarded_method(1, True)

        self.assertFalse(channel_pair.is_in_config_mode)
