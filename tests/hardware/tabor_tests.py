import unittest
import itertools
import numpy as np

from teawg import model_properties_dict

from qctoolkit.hardware.awgs.tabor import TaborException, TaborProgram, \
    TaborSegment, TaborSequencing, with_configuration_guard, PlottableProgram
from qctoolkit._program._loop import MultiChannelProgram, Loop
from qctoolkit._program.instructions import InstructionBlock
from qctoolkit.hardware.util import voltage_to_uint16

from tests.pulses.sequencing_dummies import DummyWaveform
from tests._program.loop_tests import LoopTests, WaveformGenerator, MultiChannelTests


class TaborSegmentTests(unittest.TestCase):
    def test_init(self):
        with self.assertRaises(TaborException):
            TaborSegment(None, None, None, None)
        with self.assertRaises(TaborException):
            TaborSegment(np.zeros(5), np.zeros(4), None, None)
        with self.assertRaises(TaborException):
            TaborSegment(np.zeros(4), np.zeros(4), np.zeros(4), None)
        with self.assertRaises(TaborException):
            TaborSegment(np.zeros(4), np.zeros(4), None, np.zeros(4))

        ch_a = np.asarray(100 + np.arange(6), dtype=np.uint16)
        ch_b = np.asarray(1000 + np.arange(6), dtype=np.uint16)

        marker_a = np.ones(3, dtype=bool)
        marker_b = np.arange(3, dtype=np.uint16)

        ts = TaborSegment(ch_a=ch_a, ch_b=ch_b, marker_a=marker_a, marker_b=marker_b)
        self.assertIs(ts.ch_a, ch_a)
        self.assertIs(ts.ch_b, ch_b)
        self.assertIs(ts.marker_a, marker_a)
        self.assertIsNot(ts.marker_b, marker_b)
        np.testing.assert_equal(ts.marker_b, marker_b != 0)

    def test_num_points(self):
        self.assertEqual(TaborSegment(np.zeros(6), np.zeros(6), None, None).num_points, 6)

    def test_data_a(self):
        ch_a = np.asarray(100 + np.arange(32), dtype=np.uint16)
        ch_b = np.asarray(1000 + np.arange(32), dtype=np.uint16)

        marker_a = np.ones(16, dtype=bool)
        marker_b = np.asarray(list(range(5)) + list(range(6)) + list(range(5)), dtype=np.uint16)

        on = 1 << 14
        off = 0
        marker_a_data = np.asarray([0]*8 + [on]*8 +
                                   [0]*8 + [on]*8, dtype=np.uint16)

        on = 1 << 15
        off = 0
        marker_b_data = np.asarray([0]*8 + [off] + [on]*4 + [off] + [on]*2 +
                                   [0]*8 + [on]*3 + [off] + [on]*4)

        ts = TaborSegment(ch_a=ch_a, ch_b=ch_b, marker_a=None, marker_b=None)
        self.assertIs(ts.data_a, ch_a)

        ts = TaborSegment(ch_a=ch_a, ch_b=ch_b, marker_a=marker_a, marker_b=None)
        expected_data = ch_a + marker_a_data
        np.testing.assert_equal(ts.data_a, expected_data)

        ts = TaborSegment(ch_a=ch_a, ch_b=ch_b, marker_a=None, marker_b=marker_b)
        expected_data = ch_a + marker_b_data
        np.testing.assert_equal(ts.data_a, expected_data)

        ts = TaborSegment(ch_a=ch_a, ch_b=ch_b, marker_a=marker_a, marker_b=marker_b)
        expected_data = ch_a + marker_b_data + marker_a_data
        np.testing.assert_equal(ts.data_a, expected_data)

        with self.assertRaises(NotImplementedError):
            TaborSegment(ch_a=None, ch_b=ch_b, marker_a=marker_a, marker_b=marker_b).data_a

    def test_data_b(self):
        ch_a = np.asarray(100 + np.arange(6), dtype=np.uint16)
        ch_b = np.asarray(1000 + np.arange(6), dtype=np.uint16)

        marker_a = np.ones(3, dtype=bool)
        marker_b = np.arange(3, dtype=np.uint16)

        ts = TaborSegment(ch_a=ch_a, ch_b=ch_b, marker_a=marker_a, marker_b=marker_b)

        self.assertIs(ts.data_b, ch_b)

    def test_from_binary_segment(self):
        ch_a = np.asarray(100 + np.arange(32), dtype=np.uint16)
        ch_b = np.asarray(1000 + np.arange(32), dtype=np.uint16)

        marker_a = np.ones(16, dtype=bool)
        marker_b = np.asarray(list(range(5)) + list(range(6)) + list(range(5)), dtype=np.uint16)

        segment = TaborSegment(ch_a=ch_a, ch_b=ch_b, marker_a=marker_a, marker_b=marker_b)

        binary = segment.get_as_binary()

        reconstructed = TaborSegment.from_binary_segment(binary)

        self.assertEqual(segment, reconstructed)

    def test_from_binary_data(self):
        ch_a = np.asarray(100 + np.arange(32), dtype=np.uint16)
        ch_b = np.asarray(1000 + np.arange(32), dtype=np.uint16)

        marker_a = np.ones(16, dtype=bool)
        marker_b = np.asarray(list(range(5)) + list(range(6)) + list(range(5)), dtype=np.uint16)

        segment = TaborSegment(ch_a=ch_a, ch_b=ch_b, marker_a=marker_a, marker_b=marker_b)

        data_a = segment.data_a
        data_b = segment.data_b

        reconstructed = TaborSegment.from_binary_data(data_a, data_b)

        self.assertEqual(segment, reconstructed)

    def test_eq(self):
        ch_a = np.asarray(100 + np.arange(32), dtype=np.uint16)
        ch_b = np.asarray(1000 + np.arange(32), dtype=np.uint16)

        marker_ones = np.ones(16, dtype=bool)
        marker_random = np.asarray(list(range(5)) + list(range(6)) + list(range(5)), dtype=np.uint16)
        marker_zeros = np.zeros(16, dtype=bool)

        segment_1 = TaborSegment(ch_a=ch_a, ch_b=ch_b, marker_a=marker_ones, marker_b=marker_random)
        segment_2 = TaborSegment(ch_a=ch_a, ch_b=ch_a, marker_a=marker_ones, marker_b=marker_random)

        segment_a0 = TaborSegment(ch_a=ch_a, ch_b=ch_b, marker_a=marker_zeros, marker_b=marker_random)
        segment_anone = TaborSegment(ch_a=ch_a, ch_b=ch_b, marker_a=None, marker_b=marker_random)
        segment_none = TaborSegment(ch_a=ch_a, ch_b=ch_b, marker_a=None, marker_b=None)

        self.assertEqual(segment_1, segment_1)
        self.assertNotEqual(segment_1, segment_2)

        self.assertEqual(segment_a0, segment_anone)
        self.assertEqual(segment_anone, segment_a0)
        self.assertEqual(segment_anone, segment_anone)
        self.assertNotEqual(segment_anone, segment_none)
        self.assertEqual(segment_none, segment_none)
        self.assertNotEqual(segment_a0, segment_1)


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
                yield next(alternating_on_off)[::2]
                yield np.zeros(192)[::2]

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
                np.testing.assert_equal(sampled_seg.marker_a, np.ones(192, dtype=np.uint16)[::2])
            else:
                np.testing.assert_equal(sampled_seg.marker_a, np.zeros(192, dtype=np.uint16)[::2])
            np.testing.assert_equal(sampled_seg.marker_b, np.zeros(192, dtype=np.uint16)[::2])

            np.testing.assert_equal(sampled_seg.ch_a, data[0])
            np.testing.assert_equal(sampled_seg.ch_b, data[1])


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


class PlottableProgramTests(unittest.TestCase):
    def setUp(self):
        def make_read_waveform(data):
            assert len(data) % 16 == 0

            ch_0 = []
            ch_1 = []
            for i, x in enumerate(data):
                ch_0.append(x+1000)
                ch_1.append(x)

            #ch_0.extend(ch_0[-1:]*16)
            #ch_1.extend(ch_1[-1:]*16)

            ch_0 = np.array(ch_0, dtype=np.uint16)
            ch_1 = np.array(ch_1, dtype=np.uint16)
            return np.concatenate((ch_0.reshape((-1, 16)), ch_1.reshape((-1, 16))), 1).ravel()

        self.read_waveforms = [make_read_waveform(np.arange(32)), make_read_waveform(np.arange(32, 48))]
        self.read_sequencer_tables = [(np.array([1, 1]),
                                       np.array([1, 2]),
                                       np.array([0, 0])),

                                      (np.array([1, 2, 1]),
                                       np.array([1, 2, 1]),
                                       np.array([0, 0, 0]))]
        self.read_adv_sequencer_table = (np.array([1, 1, 2]),
                                         np.array([1, 2, 1]),
                                         np.array([0, 0, 0]))

        self.waveforms = ((np.arange(32, dtype=np.uint16), np.arange(32, 48, dtype=np.uint16)),
                          (1000+np.arange(32, dtype=np.uint16), 1000+np.arange(32, 48, dtype=np.uint16)))
        self.segments = [TaborSegment.from_binary_data(a, b) for a, b in zip(*self.waveforms)]
        self.sequencer_tables = [[(1, 1, 0), (1, 2, 0)],
                                 [(1, 1, 0), (2, 2, 0), (1, 1, 0)]]
        self.adv_sequencer_table = [(1, 1, 0), (1, 2, 0), (2, 1, 0)]

    def test_init(self):
        prog = PlottableProgram(self.segments, self.sequencer_tables, self.adv_sequencer_table)
        np.testing.assert_equal(self.segments, prog._segments)
        self.assertEqual(self.sequencer_tables, prog._sequence_tables)
        self.assertEqual(self.adv_sequencer_table, prog._advanced_sequence_table)

    def test_from_read_data(self):
        prog = PlottableProgram.from_read_data(self.read_waveforms,
                                               self.read_sequencer_tables,
                                               self.read_adv_sequencer_table)
        self.assertEqual(self.segments, prog._segments)
        self.assertEqual(self.sequencer_tables, prog._sequence_tables)
        self.assertEqual(self.adv_sequencer_table, prog._advanced_sequence_table)

    def test_iter(self):
        prog = PlottableProgram(self.segments, self.sequencer_tables, self.adv_sequencer_table)

        ch = itertools.chain(range(32), range(32, 48),
                             range(32), range(32, 48), range(32, 48), range(32),
                             range(32), range(32, 48), range(32), range(32, 48))
        ch_0 = np.fromiter(ch, dtype=np.uint16)
        ch_1 = ch_0 + 1000

        for expected, found in zip(ch_0, prog.iter_samples(0, True, True)):
            self.assertEqual(expected, found)

        for expected, found in zip(ch_1, prog.iter_samples(1, True, True)):
            self.assertEqual(expected, found)

    def test_get_advanced_sequence_table(self):
        adv_seq = [(1, 1, 1)] + self.adv_sequencer_table + [(1, 1, 0)]
        prog = PlottableProgram(self.segments, self.sequencer_tables, adv_seq)

        self.assertEqual(prog._get_advanced_sequence_table(), self.adv_sequencer_table)
        self.assertEqual(prog._get_advanced_sequence_table(with_first_idle=True),
                         [(1, 1, 1)] + self.adv_sequencer_table)

        self.assertEqual(prog._get_advanced_sequence_table(with_first_idle=True, with_last_idles=True),
                         adv_seq)

    def test_builtint_conversion(self):
        prog = PlottableProgram(self.segments, self.sequencer_tables, self.adv_sequencer_table)

        prog = PlottableProgram.from_builtin(prog.to_builtin())

        np.testing.assert_equal(self.segments, prog._segments)
        self.assertEqual(self.sequencer_tables, prog._sequence_tables)
        self.assertEqual(self.adv_sequencer_table, prog._advanced_sequence_table)

    def test_eq(self):
        prog1 = PlottableProgram(self.segments, self.sequencer_tables, self.adv_sequencer_table)

        prog2 = PlottableProgram(self.segments, self.sequencer_tables, self.adv_sequencer_table)

        self.assertEqual(prog1, prog2)

    def test_get_waveforms(self):
        prog = PlottableProgram(self.segments, self.sequencer_tables, self.adv_sequencer_table)

        expected_waveforms_0 = [np.arange(32), np.arange(32, 48), np.arange(32),
                                np.arange(32, 48), np.arange(32), np.arange(32),
                                np.arange(32, 48), np.arange(32), np.arange(32, 48)]

        np.testing.assert_equal(expected_waveforms_0, prog.get_waveforms(0))

        expected_waveforms_1 = [wf + 1000 for wf in expected_waveforms_0]
        np.testing.assert_equal(expected_waveforms_1, prog.get_waveforms(1))

    def test_get_repetitions(self):
        prog = PlottableProgram(self.segments, self.sequencer_tables, self.adv_sequencer_table)

        expected_repetitions = [1, 1, 1, 2, 1, 1, 1, 1, 1]
        np.testing.assert_equal(expected_repetitions, prog.get_repetitions())

    def test_get_as_single_waveform(self):
        prog = PlottableProgram(self.segments, self.sequencer_tables, self.adv_sequencer_table)

        expected_single_waveform_0 = np.fromiter(prog.iter_samples(0), dtype=np.uint16)
        expected_single_waveform_1 = np.fromiter(prog.iter_samples(1), dtype=np.uint16)

        np.testing.assert_equal(prog.get_as_single_waveform(0), expected_single_waveform_0)
        np.testing.assert_equal(prog.get_as_single_waveform(1), expected_single_waveform_1)
