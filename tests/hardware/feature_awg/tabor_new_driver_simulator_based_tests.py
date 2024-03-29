import unittest
import subprocess
import time
import platform
import os
from typing import List, Tuple, Optional, Any

try:
    import tabor_control
except ImportError as err:
    raise unittest.SkipTest("tabor_control not present") from err
import numpy as np

from qupulse._program.tabor import TableDescription, TableEntry
from qupulse.hardware.feature_awg.features import DeviceControl, VoltageRange, ProgramManagement, SCPI, VolatileParameters
from qupulse.hardware.feature_awg.tabor import TaborDevice, TaborSegment
from qupulse.utils.types import TimeType

from tests.hardware.tabor_simulator_based_tests import TaborSimulatorManager


@unittest.skipIf(platform.system() != 'Windows', "Simulator currently only available on Windows :(")
class TaborSimulatorBasedTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.instrument: TaborDevice = None

    @classmethod
    def setUpClass(cls):
        cls.simulator_manager = TaborSimulatorManager(TaborDevice, 'instr_addr',
                                                      dict(device_name='testDevice', reset=True, paranoia_level=2),
                                                      'WX2184C.exe', os.path.dirname(__file__))
        try:
            cls.simulator_manager.start_simulator()
        except RuntimeError as err:
            raise unittest.SkipTest(*err.args) from err

    @classmethod
    def tearDownClass(cls):
        del cls.simulator_manager

    def setUp(self):
        self.instrument = self.simulator_manager.connect()

    def tearDown(self):
        self.instrument[DeviceControl].reset()
        self.simulator_manager.disconnect()

    @staticmethod
    def to_new_sequencer_tables(sequencer_tables: List[List[Tuple[int, int, int]]]
                                ) -> List[List[Tuple[TableDescription, Optional[Any]]]]:
        return [[(TableDescription(*entry), None) for entry in sequencer_table]
                for sequencer_table in sequencer_tables]

    @staticmethod
    def to_new_advanced_sequencer_table(advanced_sequencer_table: List[Tuple[int, int, int]]) -> List[TableDescription]:
        return [TableDescription(*entry) for entry in advanced_sequencer_table]


class TaborAWGRepresentationTests(TaborSimulatorBasedTest):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_sample_rate(self):
        for ch_tuple in self.instrument.channel_tuples:
            self.assertIsInstance(ch_tuple.sample_rate, TimeType)

        self.instrument[SCPI].send_cmd(':INST:SEL 1')
        self.instrument[SCPI].send_cmd(':FREQ:RAST 2.3e9')

        self.assertEqual(2300000000, self.instrument.channel_tuples[0].sample_rate)

    def test_amplitude(self):
        for channel in self.instrument.channels:
            self.assertIsInstance(channel[VoltageRange].amplitude, float)

        self.instrument[SCPI].send_cmd(':INST:SEL 1; :OUTP:COUP DC')
        self.instrument[SCPI].send_cmd(':VOLT 0.7')

        self.assertAlmostEqual(.7, self.instrument.channels[0][VoltageRange].amplitude)

    def test_select_marker(self):
        with self.assertRaises(IndexError):
            self.instrument.marker_channels[6]._select()

        self.instrument.marker_channels[1]._select()
        selected = self.instrument[SCPI].send_query(':SOUR:MARK:SEL?')
        self.assertEqual(selected, '2')

        self.instrument.marker_channels[0]._select()
        selected = self.instrument[SCPI].send_query(':SOUR:MARK:SEL?')
        self.assertEqual(selected, '1')

    def test_select_channel(self):
        with self.assertRaises(IndexError):
            self.instrument.channels[6]._select()

        self.instrument.channels[0]._select()
        self.assertEqual(self.instrument[SCPI].send_query(':INST:SEL?'), '1')

        self.instrument.channels[3]._select()
        self.assertEqual(self.instrument[SCPI].send_query(':INST:SEL?'), '4')


class TaborMemoryReadTests(TaborSimulatorBasedTest):
    def setUp(self):
        super().setUp()

        ramp_up = np.linspace(0, 2**14-1, num=192, dtype=np.uint16)
        ramp_down = ramp_up[::-1]
        zero = np.ones(192, dtype=np.uint16) * 2**13
        sine = ((np.sin(np.linspace(0, 2*np.pi, 192+64)) + 1) / 2 * (2**14 - 1)).astype(np.uint16)

        self.segments = [TaborSegment.from_sampled(ramp_up, ramp_up, None, None),
                         TaborSegment.from_sampled(ramp_down, zero, None, None),
                         TaborSegment.from_sampled(sine, sine, None, None)]

        self.zero_segment = TaborSegment.from_sampled(zero, zero, None, None)

        # program 1
        self.sequence_tables_raw = [[(10, 0, 0), (10, 1, 0), (10, 0, 0), (10, 1, 0)],
                                    [(1, 0, 0), (1, 1, 0), (1, 0, 0), (1, 1, 0)]]
        self.advanced_sequence_table = [(1, 1, 0), (1, 2, 0)]

        self.sequence_tables = self.to_new_sequencer_tables(self.sequence_tables_raw)
        self.advanced_sequence_table = self.to_new_advanced_sequencer_table(self.advanced_sequence_table)

        self.channel_pair = self.instrument.channel_tuples[0]

    def arm_program(self, sequencer_tables, advanced_sequencer_table, mode, waveform_to_segment_index):
        class DummyProgram:
            @staticmethod
            def get_sequencer_tables():
                return sequencer_tables

            @staticmethod
            def get_advanced_sequencer_table():
                return advanced_sequencer_table

            @staticmethod
            def update_volatile_parameters(parameters):
                modifications = {1: TableEntry(repetition_count=5, element_number=2, jump_flag=0),
                                 (0, 1): TableDescription(repetition_count=50, element_id=1, jump_flag=0)}
                return modifications

            markers = (None, None)
            channels = (1, 2)

            waveform_mode = mode

        self.channel_pair._known_programs['dummy_program'] = (waveform_to_segment_index, DummyProgram)
        self.channel_pair[ProgramManagement]._change_armed_program('dummy_program')

    def test_read_waveforms(self):
        self.channel_pair._amend_segments(self.segments)

        waveforms = self.channel_pair.read_waveforms()

        segments = [TaborSegment.from_binary_segment(waveform)
                    for waveform in waveforms]

        expected = [self.zero_segment, *self.segments]

        for ex, r in zip(expected, segments):
            ex1, ex2 = ex.data_a, ex.data_b
            r1, r2 = r.data_a, r.data_b
            np.testing.assert_equal(ex1, r1)
            np.testing.assert_equal(ex2, r2)

        self.assertEqual(expected, segments)

    def test_read_sequence_tables(self):
        self.channel_pair._amend_segments(self.segments)
        self.arm_program(self.sequence_tables, self.advanced_sequence_table, None, np.asarray([1, 2]))

        sequence_tables = self.channel_pair.read_sequence_tables()

        actual_sequence_tables = [self.channel_pair[ProgramManagement]._idle_sequence_table] + [[(rep, index+2, jump)
                                                                             for rep, index, jump in table]
                                                                             for table in self.sequence_tables_raw]

        expected = list(tuple(np.asarray(d)
                              for d in zip(*table))
                        for table in actual_sequence_tables)

        np.testing.assert_equal(sequence_tables, expected)

    def test_read_advanced_sequencer_table(self):
        self.channel_pair._amend_segments(self.segments)
        self.arm_program(self.sequence_tables, self.advanced_sequence_table, None, np.asarray([1, 2]))

        actual_advanced_table = [(1, 1, 0)] + [(rep, idx + 1, jmp) for rep, idx, jmp in self.advanced_sequence_table]

        expected = list(np.asarray(d)
                        for d in zip(*actual_advanced_table))

        advanced_table = self.channel_pair.read_advanced_sequencer_table()
        np.testing.assert_equal(advanced_table, expected)

    def test_set_volatile_parameter(self):
        self.channel_pair._amend_segments(self.segments)
        self.arm_program(self.sequence_tables, self.advanced_sequence_table, None, np.asarray([1, 2]))

        para = {'a': 5}
        actual_sequence_tables = [self.channel_pair[ProgramManagement]._idle_sequence_table] + [[(rep, index + 2, jump)
                                                                              for rep, index, jump in table]
                                                                             for table in self.sequence_tables_raw]

        actual_advanced_table = [(1, 1, 0)] + [(rep, idx + 1, jmp) for rep, idx, jmp in self.advanced_sequence_table]

        self.channel_pair[VolatileParameters].set_volatile_parameters('dummy_program', parameters=para)

        actual_sequence_tables[1][1] = (50, 3, 0)
        actual_advanced_table[2] = (5, 3, 0)

        sequence_table = self.channel_pair.read_sequence_tables()
        expected = list(tuple(np.asarray(d)
                              for d in zip(*table))
                        for table in actual_sequence_tables)
        np.testing.assert_equal(sequence_table, expected)

        advanced_table = self.channel_pair.read_advanced_sequencer_table()
        expected = list(np.asarray(d)
                        for d in zip(*actual_advanced_table))
        np.testing.assert_equal(advanced_table, expected)
