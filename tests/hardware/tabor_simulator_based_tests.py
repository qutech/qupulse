import unittest
import subprocess
import time

import pytabor
import numpy as np

from qctoolkit.hardware.awgs.tabor import TaborAWGRepresentation, TaborException, TaborSegment, TaborChannelPair, PlottableProgram


class TaborSimulatorBasedTest(unittest.TestCase):
    try_connecting_to_existing_simulator = True
    simulator_executable = 'WX2184C.exe'

    simulator_process = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.instrument = None

    @classmethod
    def killRunningSimulators(cls):
        subprocess.run(['Taskkill', '/IM WX2184C.exe'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    @classmethod
    def setUpClass(cls):
        if cls.try_connecting_to_existing_simulator:
            if pytabor.open_session('127.0.0.1') is not None:
                return
        cls.killRunningSimulators()

        cls.simulator_process = subprocess.Popen([cls.simulator_executable, '/switch-on', '/gui-in-tray'])

        start = time.time()
        while pytabor.open_session('127.0.0.1') is None:
            if cls.simulator_process.returncode:
                raise RuntimeError('Simulator exited with return code {}'.format(cls.simulator_process.returncode))
            if time.time() - start > 20.:
                raise RuntimeError('Could not connect to simulator')
            time.sleep(0.1)

    @classmethod
    def tearDownClass(cls):
        if cls.simulator_process is not None and not cls.try_connecting_to_existing_simulator:
            cls.simulator_process.kill()

    def setUp(self):
        self.instrument = TaborAWGRepresentation('127.0.0.1',
                                                 reset=True,
                                                 paranoia_level=2)

        if self.instrument.main_instrument.visa_inst is None:
            raise RuntimeError('Could not connect to simulator')

    def tearDown(self):
        self.instrument.reset()
        for device in self.instrument.all_devices:
            device.close()
        self.instrument = None


class TaborAWGRepresentationTests(TaborSimulatorBasedTest):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_sample_rate(self):
        for ch in (1, 2, 3, 4):
            self.assertIsInstance(self.instrument.sample_rate(ch), int)

        with self.assertRaises(TaborException):
            self.instrument.sample_rate(0)

        self.instrument.send_cmd(':INST:SEL 1;')
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

        self.segments = [TaborSegment(ramp_up, ramp_up),
                         TaborSegment(ramp_down, zero),
                         TaborSegment(sine, sine)]

        self.zero_segment = TaborSegment(zero, zero)

        # program 1
        self.sequence_tables = [[(10, 0, 0), (10, 1, 0), (10, 0, 0), (10, 1, 0)],
                                [(1, 0, 0), (1, 1, 0), (1, 0, 0), (1, 1, 0)]]
        self.advanced_sequence_table = [(1, 1, 0), (1, 2, 0)]

        self.channel_pair = TaborChannelPair(self.instrument, (1, 2), 'tabor_unit_test')

    def arm_program(self, sequencer_tables, advanced_sequencer_table, mode, waveform_to_segment_index):
        class DummyProgram:
            @staticmethod
            def get_sequencer_tables():
                return sequencer_tables

            @staticmethod
            def get_advanced_sequencer_table():
                return advanced_sequencer_table

            markers = (None, None)
            channels = (1, 2)

            waveform_mode = mode

        self.channel_pair._known_programs['dummy_program'] = (waveform_to_segment_index, DummyProgram)
        self.channel_pair.change_armed_program('dummy_program')

    def test_read_waveforms(self):
        self.channel_pair._amend_segments(self.segments)

        waveforms = self.channel_pair.read_waveforms()

        reformated = PlottableProgram._reformat_waveforms(waveforms)

        expected = [self.zero_segment, *self.segments]
        for (ex1, ex2), r1, r2 in zip(expected, *reformated):
            np.testing.assert_equal(ex1, r1)
            np.testing.assert_equal(ex2, r2)

    def test_read_sequence_tables(self):
        self.channel_pair._amend_segments(self.segments)
        self.arm_program(self.sequence_tables, self.advanced_sequence_table, None, np.asarray([1, 2]))

        sequence_tables = self.channel_pair.read_sequence_tables()

        actual_sequece_tables = [self.channel_pair._idle_sequence_table] + [[(rep, index+2, jump)
                                                                             for rep, index, jump in table]
                                                                            for table in self.sequence_tables]

        expected = list(tuple(np.asarray(d)
                              for d in zip(*table))
                        for table in actual_sequece_tables)

        np.testing.assert_equal(sequence_tables, expected)

    def test_read_advanced_sequencer_table(self):
        self.channel_pair._amend_segments(self.segments)
        self.arm_program(self.sequence_tables, self.advanced_sequence_table, None, np.asarray([1, 2]))

        actual_advanced_table = [(1, 1, 1)] + [(rep, idx+1, jmp) for rep, idx, jmp in self.advanced_sequence_table]

        expected = list(np.asarray(d)
                        for d in zip(*actual_advanced_table))

        advanced_table = self.channel_pair.read_advanced_sequencer_table()
        np.testing.assert_equal(advanced_table, expected)
