import unittest
import subprocess
import time
import platform
import os

import pytabor
import numpy as np

from qupulse.hardware.awgs.tabor import TaborAWGRepresentation, TaborChannelPair
from qupulse._program.tabor import TaborSegment, PlottableProgram, TaborException, TableDescription, TableEntry
from typing import List, Tuple, Optional, Any

class TaborSimulatorManager:
    def __init__(self,
                 simulator_executable='WX2184C.exe',
                 simulator_path=os.path.realpath(os.path.dirname(__file__))):
        self.simulator_executable = simulator_executable
        self.simulator_path = simulator_path

        self.started_simulator = False

        self.simulator_process = None
        self.instrument = None

    def kill_running_simulators(self):
        command = 'Taskkill', '/IM {simulator_executable}'.format(simulator_executable=self.simulator_executable)
        try:
            subprocess.run([command],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except FileNotFoundError:
            pass

    @property
    def simulator_full_path(self):
        return os.path.join(self.simulator_path, self.simulator_executable)

    def start_simulator(self, try_connecting_to_existing_simulator=True, max_wait_time=30):
        if try_connecting_to_existing_simulator:
            if pytabor.open_session('127.0.0.1') is not None:
                return

        if not os.path.isfile(self.simulator_full_path):
            raise RuntimeError('Cannot locate simulator executable.')

        self.kill_running_simulators()

        self.simulator_process = subprocess.Popen([self.simulator_full_path, '/switch-on', '/gui-in-tray'])

        start = time.time()
        while pytabor.open_session('127.0.0.1') is None:
            if self.simulator_process.returncode:
                raise RuntimeError('Simulator exited with return code {}'.format(self.simulator_process.returncode))
            if time.time() - start > max_wait_time:
                raise RuntimeError('Could not connect to simulator')
            time.sleep(0.1)

    def connect(self):
        self.instrument = TaborAWGRepresentation('127.0.0.1',
                                                 reset=True,
                                                 paranoia_level=2)

        if self.instrument.main_instrument.visa_inst is None:
            raise RuntimeError('Could not connect to simulator')
        return self.instrument

    def disconnect(self):
        for device in self.instrument.all_devices:
            device.close()
        self.instrument = None

    def __del__(self):
        if self.started_simulator and self.simulator_process:
            self.simulator_process.kill()


@unittest.skipIf(platform.system() != 'Windows', "Simulator currently only available on Windows :(")
class TaborSimulatorBasedTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.instrument = None

    @classmethod
    def setUpClass(cls):
        cls.simulator_manager = TaborSimulatorManager('WX2184C.exe', os.path.dirname(__file__))
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
        self.instrument.reset()
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
        for ch in (1, 2, 3, 4):
            self.assertIsInstance(self.instrument.sample_rate(ch), int)

        with self.assertRaises(TaborException):
            self.instrument.sample_rate(0)

        self.instrument.send_cmd(':INST:SEL 1')
        self.instrument.send_cmd(':FREQ:RAST 2.3e9')

        self.assertEqual(2300000000, self.instrument.sample_rate(1))

    def test_amplitude(self):
        for ch in (1, 2, 3, 4):
            self.assertIsInstance(self.instrument.amplitude(ch), float)

        with self.assertRaises(TaborException):
            self.instrument.amplitude(0)

        self.instrument.send_cmd(':INST:SEL 1; :OUTP:COUP DC')
        self.instrument.send_cmd(':VOLT 0.7')

        self.assertAlmostEqual(.7, self.instrument.amplitude(1))

    def test_select_marker(self):
        with self.assertRaises(TaborException):
            self.instrument.select_marker(6)

        self.instrument.select_marker(2)
        selected = self.instrument.send_query(':SOUR:MARK:SEL?')
        self.assertEqual(selected, '2')

        self.instrument.select_marker(1)
        selected = self.instrument.send_query(':SOUR:MARK:SEL?')
        self.assertEqual(selected, '1')

    def test_select_channel(self):
        with self.assertRaises(TaborException):
            self.instrument.select_channel(6)

        self.instrument.select_channel(1)
        self.assertEqual(self.instrument.send_query(':INST:SEL?'), '1')

        self.instrument.select_channel(4)
        self.assertEqual(self.instrument.send_query(':INST:SEL?'), '4')


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

        self.channel_pair = TaborChannelPair(self.instrument, (1, 2), 'tabor_unit_test')

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
        self.channel_pair.change_armed_program('dummy_program')

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

        actual_sequence_tables = [self.channel_pair._idle_sequence_table] + [[(rep, index+2, jump)
                                                                             for rep, index, jump in table]
                                                                             for table in self.sequence_tables_raw]

        expected = list(tuple(np.asarray(d)
                              for d in zip(*table))
                        for table in actual_sequence_tables)

        np.testing.assert_equal(sequence_tables, expected)

    def test_read_advanced_sequencer_table(self):
        self.channel_pair._amend_segments(self.segments)
        self.arm_program(self.sequence_tables, self.advanced_sequence_table, None, np.asarray([1, 2]))

        actual_advanced_table = [(1, 1, 1)] + [(rep, idx+1, jmp) for rep, idx, jmp in self.advanced_sequence_table]

        expected = list(np.asarray(d)
                        for d in zip(*actual_advanced_table))

        advanced_table = self.channel_pair.read_advanced_sequencer_table()
        np.testing.assert_equal(advanced_table, expected)

    def test_set_volatile_parameter(self):
        self.channel_pair._amend_segments(self.segments)
        self.arm_program(self.sequence_tables, self.advanced_sequence_table, None, np.asarray([1, 2]))

        para = {'a': 5}
        actual_sequence_tables = [self.channel_pair._idle_sequence_table] + [[(rep, index + 2, jump)
                                                                              for rep, index, jump in table]
                                                                             for table in self.sequence_tables_raw]

        actual_advanced_table = [(1, 1, 1)] + [(rep, idx + 1, jmp) for rep, idx, jmp in self.advanced_sequence_table]

        self.channel_pair.set_volatile_parameters('dummy_program', parameters=para)

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
