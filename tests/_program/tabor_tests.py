import unittest
import itertools
import numpy as np
from qupulse.utils.types import FrozenDict
from unittest import mock

try:
    import pytabor
except ImportError:
    pytabor = None

from teawg import model_properties_dict

from qupulse._program.tabor import TaborException, TaborProgram, \
    TaborSegment, TaborSequencing, PlottableProgram, TableDescription, make_combined_wave, TableEntry
from qupulse._program._loop import Loop
from qupulse._program.volatile import VolatileRepetitionCount
from qupulse.hardware.util import voltage_to_uint16
from qupulse.utils.types import TimeType
from qupulse.expressions import ExpressionScalar
from qupulse.parameter_scope import DictScope

from tests.pulses.sequencing_dummies import DummyWaveform
from tests._program.loop_tests import LoopTests, WaveformGenerator

from tests.hardware import dummy_modules


class PlottableProgramTests(unittest.TestCase):
    def setUp(self):
        self.ch_a = [np.arange(16, dtype=np.uint16),      np.arange(32, 64, dtype=np.uint16)]
        self.ch_b = [1000 + np.arange(16, dtype=np.uint16),  1000 + np.arange(32, 64, dtype=np.uint16)]

        self.marker_a = [np.ones(8, bool), np.array([0, 1]*8, dtype=bool)]
        self.marker_b = [np.array([0, 0, 0, 1]*2, bool), np.array([1, 0, 1, 1] * 4, dtype=bool)]

        self.segments = [TaborSegment.from_sampled(ch_a, ch_b, marker_a, marker_b)
                         for ch_a, ch_b, marker_a, marker_b in zip(self.ch_a, self.ch_b, self.marker_a, self.marker_b)]

        self.sequencer_tables = [[(1, 1, 0), (1, 2, 0)],
                                 [(1, 1, 0), (2, 2, 0), (1, 1, 0)]]
        self.adv_sequencer_table = [(1, 1, 0), (1, 2, 0), (2, 1, 0)]

        self.read_segments = [segment.get_as_binary() for segment in self.segments]
        self.read_sequencer_tables = [(np.array([1, 1]),
                                       np.array([1, 2]),
                                       np.array([0, 0])),

                                      (np.array([1, 2, 1]),
                                       np.array([1, 2, 1]),
                                       np.array([0, 0, 0]))]
        self.read_adv_sequencer_table = (np.array([1, 1, 2]),
                                         np.array([1, 2, 1]),
                                         np.array([0, 0, 0]))

        self.selection_order = [0, 1,
                                0, 1, 1, 0,
                                0, 1, 0, 1]
        self.selection_order_without_repetition = [0, 1,
                                                   0, 1, 0,
                                                   0, 1, 0, 1]

    def test_init(self):
        prog = PlottableProgram(self.segments, self.sequencer_tables, self.adv_sequencer_table)
        np.testing.assert_equal(self.segments, prog._segments)
        self.assertEqual(self.sequencer_tables, prog._sequence_tables)
        self.assertEqual(self.adv_sequencer_table, prog._advanced_sequence_table)

    def test_from_read_data(self):
        prog = PlottableProgram.from_read_data(self.read_segments,
                                               self.read_sequencer_tables,
                                               self.read_adv_sequencer_table)
        self.assertEqual(self.segments, prog._segments)
        self.assertEqual(self.sequencer_tables, prog._sequence_tables)
        self.assertEqual(self.adv_sequencer_table, prog._advanced_sequence_table)

    def test_iter(self):
        prog = PlottableProgram(self.segments, self.sequencer_tables, self.adv_sequencer_table)

        ch = itertools.chain.from_iterable(self.ch_a[idx] for idx in self.selection_order)
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

        # omit first wave
        expected_waveforms_0 = [self.ch_a[idx] for idx in self.selection_order_without_repetition]
        expected_waveforms_1 = [self.ch_b[idx] for idx in self.selection_order_without_repetition]

        np.testing.assert_equal(expected_waveforms_0, prog.get_waveforms(0))
        np.testing.assert_equal(expected_waveforms_1, prog.get_waveforms(1))

        expected_waveforms_0_marker = [self.segments[idx].data_a for idx in self.selection_order_without_repetition]
        expected_waveforms_1_marker = [self.segments[idx].data_b for idx in self.selection_order_without_repetition]

        np.testing.assert_equal(expected_waveforms_0_marker, prog.get_waveforms(0, with_marker=True))
        np.testing.assert_equal(expected_waveforms_1_marker, prog.get_waveforms(1, with_marker=True))

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


class TaborProgramTests(unittest.TestCase):
    """This test looks very messy because it was adapted multiple times to code changes while trying to keep the
    test-case similar."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setUp(self) -> None:
        self.instr_props = model_properties_dict['WX2184C'].copy()
        self.program_entry_kwargs = dict(amplitudes=(1., 1.),
                                         offsets=(0., 0.),
                                         voltage_transformations=(mock.Mock(wraps=lambda x: x),
                                                                  mock.Mock(wraps=lambda x: x)),
                                         sample_rate=TimeType.from_fraction(192, 1),
                                         mode=None
                                         )

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
                                                         duration_generator=itertools.repeat(1)))

    def test_init(self):
        prog = self.root_loop
        tabor_program = TaborProgram(prog, self.instr_props, ('A', None), (None, None), **self.program_entry_kwargs)

        self.assertEqual(tabor_program.channels, ('A', None))
        self.assertEqual(tabor_program.markers, (None, None))
        self.assertIs(prog, tabor_program.program)
        self.assertIs(self.instr_props, tabor_program._device_properties)
        self.assertEqual(frozenset('A'), tabor_program._used_channels)
        self.assertEqual(TaborSequencing.ADVANCED, tabor_program._mode)

        with self.assertRaises(KeyError):
            # C not in prog
            TaborProgram(prog, self.instr_props, ('A', 'C'), (None, None), **self.program_entry_kwargs)

        with self.assertRaises(TaborException):
            TaborProgram(prog, self.instr_props, ('A', 'B'), (None, None, None), **self.program_entry_kwargs)
        with self.assertRaises(TaborException):
            TaborProgram(prog, self.instr_props, ('A', 'B', 'C'), (None, None), **self.program_entry_kwargs)

    def test_depth_0_single_waveform(self):
        program = Loop(waveform=DummyWaveform(defined_channels={'A'}, duration=1), repetition_count=3)

        t_program = TaborProgram(program, channels=(None, 'A'), markers=(None, None),
                                 device_properties=self.instr_props, **self.program_entry_kwargs)

        self.assertEqual(t_program.waveform_mode, TaborSequencing.SINGLE)

        self.assertEqual(t_program.get_sequencer_tables(), [[(TableDescription(3, 0, 0), None)]])
        self.assertEqual(t_program.get_advanced_sequencer_table(), [TableDescription(1, 1, 0)])

    def test_depth_1_single_waveform(self):
        program = Loop(children=[Loop(waveform=DummyWaveform(defined_channels={'A'}, duration=1), repetition_count=3)],
                       repetition_count=1)

        t_program = TaborProgram(program, channels=(None, 'A'), markers=(None, None),
                                 device_properties=self.instr_props, **self.program_entry_kwargs)

        self.assertEqual(t_program.waveform_mode, TaborSequencing.SINGLE)

        self.assertEqual(t_program.get_sequencer_tables(), [[(TableDescription(3, 0, 0), None)]])
        self.assertEqual(t_program.get_advanced_sequencer_table(), [TableDescription(1, 1, 0)])

    def test_depth_1_single_sequence(self):
        program = Loop(children=[Loop(waveform=DummyWaveform(defined_channels={'A'}, duration=1), repetition_count=3),
                                 Loop(waveform=DummyWaveform(defined_channels={'A'}, duration=1), repetition_count=4)],
                       repetition_count=1)

        t_program = TaborProgram(program, channels=(None, 'A'), markers=(None, None),
                                 device_properties=self.instr_props, **self.program_entry_kwargs)

        self.assertEqual(t_program.waveform_mode, TaborSequencing.SINGLE)

        self.assertEqual(t_program.get_sequencer_tables(), [[(TableDescription(3, 0, 0), None),
                                                             (TableDescription(4, 1, 0), None)]])
        self.assertEqual(t_program.get_advanced_sequencer_table(), [TableDescription(1, 1, 0)])

    def test_depth_1_single_sequence_2(self):
        """Use the same wf twice"""
        wf_1 = DummyWaveform(defined_channels={'A'}, duration=1)
        wf_2 = DummyWaveform(defined_channels={'A'}, duration=1)

        program = Loop(children=[Loop(waveform=wf_1, repetition_count=3),
                                 Loop(waveform=wf_2, repetition_count=4),
                                 Loop(waveform=wf_1, repetition_count=1)],
                       repetition_count=1)

        t_program = TaborProgram(program, channels=(None, 'A'), markers=(None, None),
                                 device_properties=self.instr_props, **self.program_entry_kwargs)

        self.assertEqual(t_program.waveform_mode, TaborSequencing.SINGLE)

        self.assertEqual(t_program.get_sequencer_tables(), [[(TableDescription(3, 0, 0), None),
                                                             (TableDescription(4, 1, 0), None),
                                                             (TableDescription(1, 0, 0), None)]])
        self.assertEqual(t_program.get_advanced_sequencer_table(), [TableDescription(1, 1, 0)])

    def test_depth_1_advanced_sequence_unroll(self):
        wf_1 = DummyWaveform(defined_channels={'A'}, duration=1)
        wf_2 = DummyWaveform(defined_channels={'A'}, duration=1)

        program = Loop(children=[Loop(waveform=wf_1, repetition_count=3),
                                 Loop(waveform=wf_2, repetition_count=4)],
                       repetition_count=5)

        t_program = TaborProgram(program, channels=(None, 'A'), markers=(None, None),
                                 device_properties=self.instr_props, **self.program_entry_kwargs)

        self.assertEqual(t_program.waveform_mode, TaborSequencing.ADVANCED)

        # partial unroll of the last segment
        self.assertEqual(t_program.get_sequencer_tables(), [[(TableDescription(3, 0, 0), None),
                                                             (TableDescription(3, 1, 0), None),
                                                             (TableDescription(1, 1, 0), None)]])
        self.assertEqual(t_program.get_advanced_sequencer_table(), [TableEntry(5, 1, 0)])

    def test_depth_1_advanced_sequence(self):
        wf_1 = DummyWaveform(defined_channels={'A'}, duration=1)
        wf_2 = DummyWaveform(defined_channels={'A'}, duration=1)

        program = Loop(children=[Loop(waveform=wf_1, repetition_count=3),
                                 Loop(waveform=wf_2, repetition_count=4),
                                 Loop(waveform=wf_1, repetition_count=1)],
                       repetition_count=5)

        t_program = TaborProgram(program, channels=(None, 'A'), markers=(None, None),
                                 device_properties=self.instr_props, **self.program_entry_kwargs)

        self.assertEqual(t_program.waveform_mode, TaborSequencing.ADVANCED)

        # partial unroll of the last segment
        self.assertEqual(t_program.get_sequencer_tables(), [[(TableDescription(3, 0, 0), None),
                                                             (TableDescription(4, 1, 0), None),
                                                             (TableDescription(1, 0, 0), None)]])
        self.assertEqual(t_program.get_advanced_sequencer_table(), [TableEntry(5, 1, 0)])

    def test_advanced_sequence_exceptions(self):
        temp_properties = self.instr_props.copy()
        temp_properties['max_seq_len'] = 5

        program = Loop(children=[Loop(waveform=DummyWaveform(defined_channels={'A'}, duration=1), repetition_count=1)
                                 for _ in range(temp_properties['max_seq_len']+1)],
                       repetition_count=2)
        with self.assertRaises(TaborException):
            TaborProgram(program.copy_tree_structure(), channels=(None, 'A'), markers=(None, None),
                         device_properties=temp_properties, **self.program_entry_kwargs)

        temp_properties['min_seq_len'] = 100
        temp_properties['max_seq_len'] = 120
        with self.assertRaises(TaborException) as exception:
            TaborProgram(program.copy_tree_structure(), channels=(None, 'A'), markers=(None, None),
                         device_properties=temp_properties, **self.program_entry_kwargs)
        self.assertEqual(str(exception.exception), 'The algorithm is not smart enough '
                                                   'to make this sequence table longer')

        program = Loop(children=[Loop(children=[Loop(waveform=DummyWaveform(defined_channels={'A'}, duration=1)),
                                                Loop(waveform=DummyWaveform(defined_channels={'A'}, duration=1))]),
                                 Loop(children=[Loop(waveform=DummyWaveform(defined_channels={'A'}, duration=1)),
                                                Loop(waveform=DummyWaveform(defined_channels={'A'}, duration=1))])
                                 ])
        with self.assertRaises(TaborException) as exception:
            TaborProgram(program.copy_tree_structure(), channels=(None, 'A'), markers=(None, None),
                         device_properties=temp_properties, **self.program_entry_kwargs)
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

        with self.assertRaisesRegex(ValueError, "non integer length"):
            root_loop = LoopTests.get_test_loop(WaveformGenerator(
                waveform_data_generator=my_gen(self.waveform_data_generator),
                duration_generator=itertools.repeat(1 / 200),
                num_channels=4))

            TaborProgram(root_loop, self.instr_props, ('A', 'B'), (None, None), **self.program_entry_kwargs)

        root_loop = LoopTests.get_test_loop(WaveformGenerator(
            waveform_data_generator=my_gen(self.waveform_data_generator),
            duration_generator=itertools.repeat(1),
            num_channels=4))

        prog = TaborProgram(root_loop, self.instr_props, ('A', 'B'), (None, None), **self.program_entry_kwargs)

        sampled, sampled_length = prog.get_sampled_segments()

        self.assertEqual(len(sampled), 3)

        prog = TaborProgram(root_loop, self.instr_props, ('A', 'B'), ('C', None), **self.program_entry_kwargs)
        sampled, sampled_length = prog.get_sampled_segments()
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

    def test_update_volatile_parameters_with_depth1(self):
        parameters = {'s': 10, 'not': 13}
        s = VolatileRepetitionCount(expression=ExpressionScalar('s'), scope=DictScope(values=FrozenDict(s=3),
                                                                                      volatile=set('s')))

        wf_1 = DummyWaveform(defined_channels={'A'}, duration=1)
        wf_2 = DummyWaveform(defined_channels={'A'}, duration=1)

        program = Loop(children=[Loop(waveform=wf_1, repetition_count=s),
                                 Loop(waveform=wf_2, repetition_count=4),
                                 Loop(waveform=wf_1, repetition_count=1)],
                       repetition_count=1)

        t_program = TaborProgram(program, channels=(None, 'A'), markers=(None, None),
                                 device_properties=self.instr_props, **self.program_entry_kwargs)

        self.assertEqual(t_program.get_sequencer_tables(), [[(TableDescription(3, 0, 0), s.volatile_property),
                                                             (TableDescription(4, 1, 0), None),
                                                             (TableDescription(1, 0, 0), None)]])
        self.assertEqual(t_program.get_advanced_sequencer_table(), [TableDescription(1, 1, 0)])

        modifications = t_program.update_volatile_parameters(parameters)

        expected_seq = VolatileRepetitionCount(expression=ExpressionScalar('s'), scope=DictScope(values=FrozenDict(s=10), volatile=set('s')))
        expected_modifications = {(0, 0): TableDescription(10, 0, 0)}

        self.assertEqual(t_program.get_sequencer_tables(), [[(TableDescription(10, 0, 0), expected_seq.volatile_property),
                                                             (TableDescription(4, 1, 0), None),
                                                             (TableDescription(1, 0, 0), None)]])
        self.assertEqual(t_program.get_advanced_sequencer_table(), [TableDescription(1, 1, 0)])
        self.assertEqual(modifications, expected_modifications)

    def test_update_volatile_parameters_with_depth2(self):
        parameters = {'s': 10, 'a': 2, 'not': 13}
        s = VolatileRepetitionCount(expression=ExpressionScalar('s'),
                                    scope=DictScope(values=FrozenDict(s=3), volatile=set('s')))
        a = VolatileRepetitionCount(expression=ExpressionScalar('a'),
                                    scope=DictScope(values=FrozenDict(a=5), volatile=set('a')))

        wf_1 = DummyWaveform(defined_channels={'A'}, duration=1)
        wf_2 = DummyWaveform(defined_channels={'A'}, duration=1)

        program = Loop(children=[Loop(children=[Loop(waveform=wf_1, repetition_count=s),
                                                Loop(waveform=wf_2, repetition_count=4),
                                                Loop(waveform=wf_1, repetition_count=2)],
                                      repetition_count=4),
                                 Loop(children=[Loop(waveform=wf_2, repetition_count=5),
                                                Loop(waveform=wf_1, repetition_count=s),
                                                Loop(waveform=wf_2, repetition_count=5)],
                                      repetition_count=a)],
                       repetition_count=1)

        t_program = TaborProgram(program, channels=(None, 'A'), markers=(None, None),
                                 device_properties=self.instr_props, **self.program_entry_kwargs)

        self.assertEqual(t_program.get_sequencer_tables(), [[(TableDescription(3, 0, 0), s.volatile_property),
                                                             (TableDescription(4, 1, 0), None),
                                                             (TableDescription(2, 0, 0), None)],
                                                            [(TableDescription(5, 1, 0), None),
                                                             (TableDescription(3, 0, 0), s.volatile_property),
                                                             (TableDescription(5, 1, 0), None)]
                                                            ])
        self.assertEqual(t_program.get_advanced_sequencer_table(), [TableEntry(4, 1, 0), TableEntry(5, 2, 0)])

        modifications = t_program.update_volatile_parameters(parameters)

        expected_seq = VolatileRepetitionCount(expression=ExpressionScalar('s'),
                                               scope=DictScope(values=FrozenDict(s=10), volatile=set('s')))
        expected_modifications = {(0, 0): TableDescription(10, 0, 0), (1, 1): TableDescription(10, 0, 0),
                                  1: TableEntry(2, 2, 0)}

        self.assertEqual(t_program.get_sequencer_tables(), [[(TableDescription(10, 0, 0), expected_seq.volatile_property),
                                                             (TableDescription(4, 1, 0), None),
                                                             (TableDescription(2, 0, 0), None)],
                                                            [(TableDescription(5, 1, 0), None),
                                                             (TableDescription(10, 0, 0), expected_seq.volatile_property),
                                                             (TableDescription(5, 1, 0), None)]
                                                            ])
        self.assertEqual(t_program.get_advanced_sequencer_table(), [TableEntry(4, 1, 0), TableEntry(2, 2, 0)])
        self.assertEqual(modifications, expected_modifications)


class TaborSegmentTests(unittest.TestCase):
    @staticmethod
    def assert_from_sampled_consistent(ch_a, ch_b, marker_a, marker_b):
        ts = TaborSegment.from_sampled(ch_a=ch_a, ch_b=ch_b, marker_a=marker_a, marker_b=marker_b)
        np.testing.assert_equal(ts.ch_a, ch_a)
        np.testing.assert_equal(ts.ch_b, ch_b)
        np.testing.assert_equal(ts.marker_a, marker_a != 0)
        np.testing.assert_equal(ts.marker_b, marker_b != 0)
        return ts

    def test_from_sampled(self):
        with self.assertRaisesRegex(TaborException, 'Empty'):
            TaborSegment.from_sampled(None, None, None, None)
        with self.assertRaisesRegex(TaborException, 'same length'):
            TaborSegment.from_sampled(np.zeros(16, dtype=np.uint16), np.zeros(32, dtype=np.uint16), None, None)

        ch_a = np.asarray(100 + np.arange(192), dtype=np.uint16)
        ch_b = np.asarray(1000 + np.arange(192), dtype=np.uint16)

        marker_a = np.ones(192 // 2, dtype=bool)
        marker_b = np.arange(192 // 2, dtype=np.uint16)

        self.assert_from_sampled_consistent(ch_a=ch_a, ch_b=ch_b, marker_a=marker_a, marker_b=marker_b)

    def test_num_points(self):
        self.assertEqual(TaborSegment.from_sampled(np.zeros(32, dtype=np.uint16), np.zeros(32, dtype=np.uint16),
                                                   None, None).num_points, 32)

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

        ts = TaborSegment.from_sampled(ch_a=ch_a, ch_b=ch_b, marker_a=None, marker_b=None)
        np.testing.assert_equal(ts.data_a, ch_a)

        ts = TaborSegment.from_sampled(ch_a=ch_a, ch_b=ch_b, marker_a=marker_a, marker_b=None)
        expected_data = ch_a + marker_a_data
        np.testing.assert_equal(ts.data_a, expected_data)

        ts = TaborSegment.from_sampled(ch_a=ch_a, ch_b=ch_b, marker_a=None, marker_b=marker_b)
        expected_data = ch_a + marker_b_data
        np.testing.assert_equal(ts.data_a, expected_data)

        ts = TaborSegment.from_sampled(ch_a=ch_a, ch_b=ch_b, marker_a=marker_a, marker_b=marker_b)
        expected_data = ch_a + marker_b_data + marker_a_data
        np.testing.assert_equal(ts.data_a, expected_data)

    def test_data_b(self):
        ch_a = np.asarray(100 + np.arange(16), dtype=np.uint16)
        ch_b = np.asarray(1000 + np.arange(16), dtype=np.uint16)

        marker_a = np.ones(8, dtype=bool)
        marker_b = np.arange(8, dtype=np.uint16)

        ts = self.assert_from_sampled_consistent(ch_a=ch_a, ch_b=ch_b, marker_a=marker_a, marker_b=marker_b)
        np.testing.assert_equal(ts.ch_b, ts.data_b)

    def test_from_binary_segment(self):
        ch_a = np.asarray(100 + np.arange(32), dtype=np.uint16)
        ch_b = np.asarray(1000 + np.arange(32), dtype=np.uint16)

        marker_a = np.ones(16, dtype=bool)
        marker_b = np.asarray(list(range(5)) + list(range(6)) + list(range(5)), dtype=np.uint16)

        segment = TaborSegment.from_sampled(ch_a=ch_a, ch_b=ch_b, marker_a=marker_a, marker_b=marker_b)

        binary = segment.get_as_binary()

        reconstructed = TaborSegment.from_binary_segment(binary)

        self.assertEqual(segment, reconstructed)

    def test_from_binary_data(self):
        ch_a = np.asarray(100 + np.arange(32), dtype=np.uint16)
        ch_b = np.asarray(1000 + np.arange(32), dtype=np.uint16)

        marker_a = np.ones(16, dtype=bool)
        marker_b = np.asarray(list(range(5)) + list(range(6)) + list(range(5)), dtype=np.uint16)

        segment = TaborSegment.from_sampled(ch_a=ch_a, ch_b=ch_b, marker_a=marker_a, marker_b=marker_b)

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

        segment_1 = TaborSegment.from_sampled(ch_a=ch_a, ch_b=ch_b, marker_a=marker_ones, marker_b=marker_random)
        segment_2 = TaborSegment.from_sampled(ch_a=ch_a, ch_b=ch_a, marker_a=marker_ones, marker_b=marker_random)

        segment_a0 = TaborSegment.from_sampled(ch_a=ch_a, ch_b=ch_b, marker_a=marker_zeros, marker_b=marker_random)
        segment_anone = TaborSegment.from_sampled(ch_a=ch_a, ch_b=ch_b, marker_a=None, marker_b=marker_random)
        segment_none = TaborSegment.from_sampled(ch_a=ch_a, ch_b=ch_b, marker_a=None, marker_b=None)

        self.assertEqual(segment_1, segment_1)
        self.assertNotEqual(segment_1, segment_2)

        self.assertEqual(segment_a0, segment_anone)
        self.assertEqual(segment_anone, segment_a0)
        self.assertEqual(segment_anone, segment_anone)
        self.assertNotEqual(segment_anone, segment_none)
        self.assertEqual(segment_none, segment_none)
        self.assertNotEqual(segment_a0, segment_1)

        all_segments = [segment_1, segment_2, segment_a0, segment_anone, segment_none]
        for seg_a, seg_b in itertools.product(all_segments, all_segments):
            if seg_a == seg_b:
                self.assertEqual(hash(seg_a), hash(seg_b))


class TaborMakeCombinedTest(unittest.TestCase):
    @staticmethod
    def validate_result(tabor_segments, result, fill_value=None):
        pos = 0
        for i, tabor_segment in enumerate(tabor_segments):
            if i > 0:
                if tabor_segment.ch_b is None:
                    if fill_value:
                        np.testing.assert_equal(result[pos:pos + 16],
                                                np.full(16, fill_value=fill_value, dtype=np.uint16))
                else:
                    np.testing.assert_equal(result[pos:pos + 16], np.full(16, tabor_segment.ch_b[0], dtype=np.uint16))
                pos += 16

                if tabor_segment.ch_a is None:
                    if fill_value:
                        np.testing.assert_equal(result[pos:pos + 16],
                                                np.full(16, fill_value=fill_value, dtype=np.uint16))
                else:
                    np.testing.assert_equal(result[pos:pos + 16], np.full(16, tabor_segment.ch_a[0], dtype=np.uint16))
                pos += 16

            for j in range(tabor_segment.num_points // 16):
                if tabor_segment.ch_b is None:
                    if fill_value:
                        np.testing.assert_equal(result[pos:pos + 16],
                                                np.full(16, fill_value=fill_value, dtype=np.uint16))
                else:
                    np.testing.assert_equal(result[pos:pos + 16], tabor_segment.ch_b[j * 16: (j + 1) * 16])
                pos += 16

                if tabor_segment.ch_a is None:
                    if fill_value:
                        np.testing.assert_equal(result[pos:pos + 16],
                                                np.full(16, fill_value=fill_value, dtype=np.uint16))
                else:
                    np.testing.assert_equal(result[pos:pos + 16], tabor_segment.ch_a[j * 16: (j + 1) * 16])
                pos += 16

    def exec_general(self, data_1, data_2):
        tabor_segments = [TaborSegment.from_sampled(d1, d2, None, None) for d1, d2 in zip(data_1, data_2)]
        expected_length = (sum(segment.num_points for segment in tabor_segments) + 16 * (len(tabor_segments) - 1)) * 2

        result = make_combined_wave(tabor_segments)
        self.assertEqual(len(result), expected_length)

        self.validate_result(tabor_segments, result)

        destination_array = np.empty(expected_length, dtype=np.uint16)
        result = make_combined_wave(tabor_segments, destination_array=destination_array)
        self.validate_result(tabor_segments, result)
        self.assertEqual(destination_array.data, result.data)

    def test_make_comb_both(self):
        gen = itertools.count()
        data_1 = [np.fromiter(gen, count=32, dtype=np.uint16),
                  np.fromiter(gen, count=16, dtype=np.uint16),
                  np.fromiter(gen, count=192, dtype=np.uint16)]

        data_2 = [np.fromiter(gen, count=32, dtype=np.uint16),
                  np.fromiter(gen, count=16, dtype=np.uint16),
                  np.fromiter(gen, count=192, dtype=np.uint16)]
        for d in data_2:
            d += 1000

        self.exec_general(data_1, data_2)

    def test_make_single_chan(self):
        gen = itertools.count()
        data_1 = [np.fromiter(gen, count=32, dtype=np.uint16),
                  np.fromiter(gen, count=16, dtype=np.uint16),
                  np.fromiter(gen, count=192, dtype=np.uint16)]

        data_2 = [None]*len(data_1)
        self.exec_general(data_1, data_2)
        self.exec_general(data_2, data_1)

    def test_empty_segment_list(self):
        combined = make_combined_wave([])

        self.assertIsInstance(combined, np.ndarray)
        self.assertIs(combined.dtype, np.dtype('uint16'))
        self.assertEqual(len(combined), 0)


@unittest.skipIf(pytabor in (dummy_modules.dummy_pytabor, None), "Cannot compare to pytabor results")
class TaborMakeCombinedPyTaborCompareTest(TaborMakeCombinedTest):
    def exec_general(self, data_1, data_2, fill_value=None):
        tabor_segments = [TaborSegment.from_sampled(d1, d2, None, None) for d1, d2 in zip(data_1, data_2)]
        expected_length = (sum(segment.num_points for segment in tabor_segments) + 16 * (len(tabor_segments) - 1)) * 2

        offset = 0
        pyte_result = 15000*np.ones(expected_length, dtype=np.uint16)
        for i, segment in enumerate(tabor_segments):
            offset = pytabor.make_combined_wave(segment.ch_a, segment.ch_b,
                                                dest_array=pyte_result, dest_array_offset=offset,
                                                add_idle_pts=i > 0)
        self.assertEqual(expected_length, offset)

        result = make_combined_wave(tabor_segments)
        np.testing.assert_equal(pyte_result, result)

        dest_array = 15000*np.ones(expected_length, dtype=np.uint16)
        result = make_combined_wave(tabor_segments, destination_array=dest_array)
        np.testing.assert_equal(pyte_result, result)
        # test that the destination array data is not copied
        self.assertEqual(dest_array.__array_interface__['data'],
                         result.__array_interface__['data'])

        with self.assertRaises(ValueError):
            make_combined_wave(tabor_segments, destination_array=np.ones(16))
