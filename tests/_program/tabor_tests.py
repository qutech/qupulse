import unittest
import itertools

import numpy as np

from qupulse._program.tabor import PlottableProgram, TaborSegment


class PlottableProgramTests(unittest.TestCase):
    def setUp(self):
        self.ch_a = [np.arange(16, dtype=np.uint16),      np.arange(32, 64, dtype=np.uint16)]
        self.ch_b = [1000 + np.arange(16, dtype=np.uint16),  1000 + np.arange(32, 64, dtype=np.uint16)]

        self.marker_a = [np.ones(8, bool), np.array([0, 1]*8, dtype=bool)]
        self.marker_b = [np.array([0, 0, 0, 1]*2, bool), np.array([1, 0, 1, 1] * 4, dtype=bool)]

        self.segments = [TaborSegment(ch_a, ch_b, marker_a, marker_b)
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
