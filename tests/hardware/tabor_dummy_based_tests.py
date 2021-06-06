import sys
import unittest
from unittest import mock
from unittest.mock import patch, MagicMock

from typing import List, Tuple, Optional, Any
from copy import copy, deepcopy

import numpy as np
import pyvisa.constants

from qupulse.hardware.awgs.base import AWGAmplitudeOffsetHandling
from qupulse.hardware.feature_awg.tabor import TaborProgramMemory
from qupulse.hardware.awgs.tabor import TaborAWGRepresentation, TaborChannelPair
from qupulse.utils.types import TimeType
from qupulse._program._loop import Loop
from qupulse._program.tabor import TableDescription, TimeType, TableEntry, TaborSegment, TaborProgram,\
    make_combined_wave, TaborSequencing
from qupulse._program.waveforms import ConstantWaveform

from tests.pulses.sequencing_dummies import DummyWaveform

try:
    import tabor_control.device
    import tabor_control.util
except ImportError:
    tabor_control = None


class DummyTaborProgramClass:
    def __init__(self, segments=None, segment_lengths=None,
                 sequencer_tables=None, advanced_sequencer_table=None, waveform_mode=None):
        self.program = None
        self.device_properties = None
        self.channels = None
        self.markers = None

        self.segment_lengths = segment_lengths
        self.segments = segments

        self.sequencer_tables = sequencer_tables
        self.advanced_sequencer_table = advanced_sequencer_table
        self.waveform_mode = waveform_mode

        self.created = []

    def __call__(self, program, device_properties, channels, markers):
        self.program = program
        self.device_properties = device_properties
        self.channels = channels
        self.markers = markers

        class DummyTaborProgram:
            def __init__(self, class_obj: DummyTaborProgramClass):
                self.sampled_segments_calls = []
                self.class_obj = class_obj
                self.waveform_mode = class_obj.waveform_mode
            def sampled_segments(self, sample_rate, voltage_amplitude, voltage_offset, voltage_transformation):
                self.sampled_segments_calls.append((sample_rate, voltage_amplitude, voltage_offset, voltage_transformation))
                return self.class_obj.segments, self.class_obj.segment_lengths
            def get_sequencer_tables(self):
                return self.class_obj.sequencer_tables
            def get_advanced_sequencer_table(self):
                return self.class_obj.advanced_sequencer_table
        self.created.append(DummyTaborProgram(self))
        return self.created[-1]


@unittest.skipIf(tabor_control is None, "tabor_control not present")
class TaborDummyBasedTest(unittest.TestCase):
    def setUp(self):
        self.visa_instr_main = mock.sentinel
        self.visa_instr_mirror = mock.sentinel
        self.tewx_awg_main = mock.create_autospec(tabor_control.device.TEWXAwg)
        self.tewx_awg_mirror = mock.create_autospec(tabor_control.device.TEWXAwg)
        self.tewx_awg_class = mock.create_autospec(tabor_control.device.TEWXAwg, side_effects=[self.tewx_awg_main, self.tewx_awg_mirror])

        visa_instrs = [self.visa_instr_main, self.visa_instr_mirror]
        tewx_awgs = [self.tewx_awg_main, self.tewx_awg_mirror]

        device_properties = tabor_control.util.get_device_properties('TE,WX2184C,4,3.01', '1232')

        self.tewx_awg_main.dev_properties = device_properties
        self.tewx_awg_mirror.dev_properties = device_properties

        with mock.patch('tabor_control.open_session', side_effect=visa_instrs) as open_session:
            with mock.patch('tabor_control.device.TEWXAwg', side_effect=tewx_awgs) as te_class:
                self.instrument = TaborAWGRepresentation('main_instrument',
                                                         reset=True,
                                                         paranoia_level=2,
                                                         mirror_addresses=['mirror_instrument'])

                self.assertEqual(open_session.call_args_list, [mock.call('main_instrument'), mock.call('mirror_instrument')])
                self.assertEqual(te_class.call_args_list, [mock.call(self.visa_instr_main, tabor_control.ParanoiaLevel.SYST_ERR),
                                                           mock.call(self.visa_instr_mirror, tabor_control.ParanoiaLevel.SYST_ERR)])

    def reset_instrument_logs(self):
        self.tewx_awg_main.reset_mock()
        self.tewx_awg_mirror.reset_mock()

    @property
    def awg_representation(self):
        return self.instrument

    @property
    def channel_pair(self):
        return self.awg_representation.channel_pair_AB

    def reset_instrument_logs_(self):
        for device in self.instrument.all_devices:
            device.logged_commands = []
            device._send_binary_data_calls = []
            device._download_adv_seq_table_calls = []
            device._download_sequencer_table_calls = []

    def assertAllCommandLogsEqual(self, expected_log: List):
        for device in self.instrument.all_devices:
            self.assertEqual(device.send_cmd.call_args_list, expected_log)

    def assert_numpy_calls_equal(self, function: mock.Mock, expected_calls):
        call_args_list = function.call_args_list
        self.assertEqual(len(call_args_list), len(expected_calls))
        for (*_, args, kwargs), (*_, expected_args, expected_kwargs) in zip(call_args_list, expected_calls):
            np.testing.assert_equal(expected_kwargs, kwargs)
            np.testing.assert_equal(expected_args, args)

    def assert_written_segment_lengths(self, *expected_segment_lengths):
        expected_calls = [mock.call(expected, paranoia_level=None) for expected in expected_segment_lengths]
        for device in self.instrument.all_devices:
            self.assert_numpy_calls_equal(device.write_segment_lengths, expected_calls)

    def assert_written_segment_data(self, *written_segments):
        expected_calls = [mock.call(expected, paranoia_level=None) for expected in written_segments]
        for device in self.instrument.all_devices:
            self.assert_numpy_calls_equal(device.write_segment_data, expected_calls)

    def assert_written_sequencer_tables(self, *expected_sequencer_tables):
        expected_calls = [mock.call(expected, paranoia_level=None) for expected in expected_sequencer_tables]
        for device in self.instrument.all_devices:
            self.assert_numpy_calls_equal(device.write_sequencer_table, expected_calls)

    def assert_written_advanced_sequencer_tables(self, *advanced_sequencer_tables):
        expected_calls = [mock.call(expected, paranoia_level=None) for expected in advanced_sequencer_tables]
        for device in self.instrument.all_devices:
            self.assert_numpy_calls_equal(device.write_advanced_sequencer_table, expected_calls)


class TaborAWGRepresentationDummyBasedTests(TaborDummyBasedTest):
    def test_send_cmd(self):
        self.reset_instrument_logs()

        self.instrument.send_cmd('bleh', paranoia_level=2)

        self.assertAllCommandLogsEqual([((), dict(paranoia_level=2, cmd_str='bleh'))])

        self.instrument.send_cmd('bleho')
        self.assertAllCommandLogsEqual([((), dict(paranoia_level=2, cmd_str='bleh')),
                                        ((), dict(cmd_str='bleho', paranoia_level=None))])

    def test_trigger(self):
        self.reset_instrument_logs()
        self.instrument.trigger()

        self.assertAllCommandLogsEqual([((), dict(cmd_str=':TRIG', paranoia_level=None))])

    def test_paranoia_level(self):
        self.instrument.paranoia_level = tabor_control.ParanoiaLevel.NONE
        self.assertIn(self.instrument.paranoia_level, (tabor_control.ParanoiaLevel.NONE, 0))
        self.instrument.paranoia_level = tabor_control.ParanoiaLevel.OPC
        for device in self.instrument.all_devices:
            self.assertIn(device.paranoia_level, (tabor_control.ParanoiaLevel.OPC, 1))

    def test_enable(self):
        self.reset_instrument_logs()
        self.instrument.enable()

        expected_commands = [':ENAB']
        expected_log = [((), dict(cmd_str=cmd, paranoia_level=None))
                        for cmd in expected_commands]
        self.assertAllCommandLogsEqual(expected_log)


class TaborChannelPairTests(TaborDummyBasedTest):
    @staticmethod
    def to_new_sequencer_tables(sequencer_tables: List[List[Tuple[int, int, int]]]
                                ) -> List[List[Tuple[TableDescription, Optional[Any]]]]:
        return [[(TableDescription(*entry), None) for entry in sequencer_table]
                for sequencer_table in sequencer_tables]

    @staticmethod
    def to_new_advanced_sequencer_table(advanced_sequencer_table: List[Tuple[int, int, int]]) -> List[TableDescription]:
        return [TableDescription(*entry) for entry in advanced_sequencer_table]

    def test_execute_multiple_commands_with_config_guard(self):
        channel_pair = TaborChannelPair(self.instrument, identifier='asd', channels=(1, 2))
        # prevent entering and exiting configuration mode
        channel_pair._configuration_guard_count = 2

        given_commands = [':ASEQ:DEF 2,2,5,0', ':SEQ:SEL 2', ':SEQ:DEF 1,2,10,0']
        expected_command = ':ASEQ:DEF 2,2,5,0;:SEQ:SEL 2;:SEQ:DEF 1,2,10,0'
        with mock.patch.object(channel_pair.device, 'send_cmd') as send_cmd:
            channel_pair._execute_multiple_commands_with_config_guard(given_commands)
            send_cmd.assert_called_once_with(expected_command, paranoia_level=channel_pair.internal_paranoia_level)

    def test_set_volatile_parameters(self):
        channel_pair = TaborChannelPair(self.instrument, identifier='asd', channels=(1, 2))

        parameters = {'var': 2}
        modifications = {1: TableEntry(repetition_count=5, element_number=1, jump_flag=0),
                         (0, 1): TableDescription(repetition_count=10, element_id=0, jump_flag=0)}
        invalid_modification = {1: TableEntry(repetition_count=0, element_number=1, jump_flag=0)}
        no_modifications = {}

        program_mock = mock.Mock(TaborProgram)
        program_memory = TaborProgramMemory(waveform_to_segment=np.array([1, 4]), program=program_mock)

        expected_commands = {':ASEQ:DEF 2,2,5,0', ':SEQ:SEL 2', ':SEQ:DEF 1,2,10,0'}

        channel_pair._known_programs['active_program'] = program_memory
        channel_pair._known_programs['other_program'] = program_memory
        channel_pair._current_program = 'active_program'

        with mock.patch.object(program_mock, 'update_volatile_parameters', return_value=modifications) as update_prog:
            with mock.patch.object(channel_pair, '_execute_multiple_commands_with_config_guard') as ex_com:
                #with mock.patch.object(channel_pair.device.main_instrument, 'send_query'):
                channel_pair.set_volatile_parameters('other_program', parameters)
                ex_com.assert_not_called()
                update_prog.assert_called_once_with(parameters)

                channel_pair.set_volatile_parameters('active_program', parameters)
                self.assertEqual(1, ex_com.call_count)
                actual_commands, = ex_com.call_args[0]
                self.assertEqual(expected_commands, set(actual_commands))
                self.assertEqual(len(expected_commands), len(actual_commands))

                assert update_prog.call_count == 2
                update_prog.assert_called_with(parameters)

        with mock.patch.object(program_mock, 'update_volatile_parameters', return_value=no_modifications) as update_prog:
            with mock.patch.object(channel_pair, '_execute_multiple_commands_with_config_guard') as ex_com:
                channel_pair.set_volatile_parameters('active_program', parameters)

                ex_com.assert_not_called()
                update_prog.assert_called_once_with(parameters)

        with mock.patch.object(program_mock, 'update_volatile_parameters', return_value=invalid_modification) as update_prog:
            with mock.patch.object(channel_pair, '_execute_multiple_commands_with_config_guard') as ex_com:
                with self.assertRaises(ValueError):
                    channel_pair.set_volatile_parameters('active_program', parameters)

                ex_com.assert_not_called()
                update_prog.assert_called_once_with(parameters)

    def test_copy(self):
        channel_pair = TaborChannelPair(self.instrument, identifier='asd', channels=(1, 2))
        with self.assertRaises(NotImplementedError):
            copy(channel_pair)
        with self.assertRaises(NotImplementedError):
            deepcopy(channel_pair)

    def test_init(self):
        with self.assertRaises(ValueError):
            TaborChannelPair(self.instrument, identifier='asd', channels=(1, 3))

    def test_free_program(self):
        channel_pair = TaborChannelPair(self.instrument, identifier='asd', channels=(1, 2))

        with self.assertRaises(KeyError):
            channel_pair.free_program('test')

        program = TaborProgramMemory(np.array([1, 2], dtype=np.int64), None)

        channel_pair._segment_references = np.array([1, 3, 1, 0])
        channel_pair._known_programs['test'] = program
        self.assertIs(channel_pair.free_program('test'), program)

        np.testing.assert_equal(channel_pair._segment_references, np.array([1, 2, 0, 0]))

    def test_upload_exceptions(self):

        wv = ConstantWaveform(channel=1, duration=192, amplitude=0.1)

        channel_pair = TaborChannelPair(self.instrument, identifier='asd', channels=(1, 2))

        program = Loop(waveform=wv)
        with self.assertRaises(ValueError):
            channel_pair.upload('test', program, (1, 2, 3), (5, 6), (lambda x: x, lambda x: x))
        with self.assertRaises(ValueError):
            channel_pair.upload('test', program, (1, 2), (5, 6, 'a'), (lambda x: x, lambda x: x))
        with self.assertRaises(ValueError):
            channel_pair.upload('test', program, (1, 2), (3, 4), (lambda x: x,))

        old = channel_pair._amplitude_offset_handling
        with self.assertRaises(ValueError):
            channel_pair._amplitude_offset_handling = 'invalid'
            channel_pair.upload('test', program, (1, None), (None, None), (lambda x: x, lambda x: x))
        channel_pair._amplitude_offset_handling = old

        channel_pair._known_programs['test'] = TaborProgramMemory(np.array([0]), None)
        with self.assertRaises(ValueError):
            channel_pair.upload('test', program, (1, 2), (3, 4), (lambda x: x, lambda x: x))

    def test_upload(self):
        segments = np.array([1, 2, 3, 4, 5])
        segment_lengths = np.array([0, 16, 0, 16, 0], dtype=np.uint16).tolist()

        segment_references = np.array([1, 1, 2, 0, 1], dtype=np.uint32)

        w2s = np.array([-1, -1, 1, 2, -1], dtype=np.int64)
        ta = np.array([True, False, False, False, True])
        ti = np.array([-1, 3, -1, -1, -1])

        channels = (1, None)
        markers = (None, None)
        voltage_transformations = (lambda x: x, lambda x: x)
        sample_rate = TimeType.from_fraction(1, 1)

        with mock.patch('qupulse.hardware.awgs.tabor.TaborProgram', specs=TaborProgram) as DummyTaborProgram:
            tabor_program = DummyTaborProgram.return_value
            tabor_program.get_sampled_segments.return_value = (segments, segment_lengths)

            program = Loop(waveform=DummyWaveform(duration=192))

            channel_pair = TaborChannelPair(self.instrument, identifier='asd', channels=(1, 2))
            channel_pair._segment_references = segment_references

            def dummy_find_place(segments_, segement_lengths_):
                self.assertIs(segments_, segments)
                self.assertIs(segment_lengths, segement_lengths_)
                return w2s, ta, ti

            def dummy_upload_segment(segment_index, segment):
                self.assertEqual(segment_index, 3)
                self.assertEqual(segment, 2)

            def dummy_amend_segments(segments_):
                np.testing.assert_equal(segments_, np.array([1, 5]))
                return np.array([5, 6], dtype=np.int64)

            self.instrument.amplitude = mock.Mock(return_value=1.)
            self.instrument.sample_rate = mock.Mock(return_value=10**9)

            channel_pair._find_place_for_segments_in_memory = dummy_find_place
            channel_pair._upload_segment = dummy_upload_segment
            channel_pair._amend_segments = dummy_amend_segments

            channel_pair.upload('test', program, channels, markers, voltage_transformations)

            DummyTaborProgram.assert_called_once_with(
                program,
                channels=tuple(channels),
                markers=markers,
                device_properties=channel_pair.device.dev_properties,
                sample_rate=sample_rate,
                amplitudes=(.5, .5),
                offsets=(0., 0.),
                voltage_transformations=voltage_transformations
            )

            self.assertEqual(
                self.instrument.amplitude.call_args_list,
                [mock.call(1), mock.call(2)]
            )
            self.instrument.sample_rate.assert_called_once_with(1)

            # the other references are increased in amend and upload segment method
            np.testing.assert_equal(channel_pair._segment_references, np.array([1, 2, 3, 0, 1]))

            self.assertEqual(len(channel_pair._known_programs), 1)
            np.testing.assert_equal(channel_pair._known_programs['test'].waveform_to_segment,
                                    np.array([5, 3, 1, 2, 6], dtype=np.int64))

    def test_upload_offset_handling(self):

        program = Loop(waveform=ConstantWaveform(channel=1, duration=192, amplitude=0.1))

        channel_pair = TaborChannelPair(self.instrument, identifier='asd', channels=(1, 2))

        channels = (1, None)
        markers = (None, None)

        tabor_program_kwargs = dict(
            channels=channels,
            markers=markers,
            device_properties=channel_pair.device.dev_properties)

        amplitudes = (0.5, 0.3)

        test_sample_rate = TimeType.from_fraction(1, 1)
        test_amplitudes = (0.5 / 2, 0.3 / 2)
        test_offset = 0.1
        test_transform = (lambda x: x, lambda x: x)

        with patch('qupulse.hardware.awgs.tabor.TaborProgram', wraps=TaborProgram) as tabor_program_mock:
            with patch.object(self.instrument, 'offset', return_value=test_offset) as offset_mock:
                tabor_program_mock.get_sampled_segments = mock.Mock(wraps=tabor_program_mock.get_sampled_segments)

                self.instrument.amplitude = mock.Mock(side_effect=amplitudes)
                self.instrument.sample_rate = mock.Mock(return_value=10 ** 9)

                channel_pair.amplitude_offset_handling = AWGAmplitudeOffsetHandling.CONSIDER_OFFSET
                channel_pair.upload('test1', program, channels, markers, test_transform)

                tabor_program_mock.assert_called_once_with(program, **tabor_program_kwargs,
                                                           sample_rate=test_sample_rate,
                                                           amplitudes=test_amplitudes,
                                                           offsets=(test_offset, test_offset),
                                                           voltage_transformations=test_transform)
                self.assertEqual([mock.call(1), mock.call(2)], offset_mock.call_args_list)
                offset_mock.reset_mock()
                tabor_program_mock.reset_mock()

                self.instrument.amplitude = mock.Mock(side_effect=amplitudes)
                self.instrument.sample_rate = mock.Mock(return_value=10 ** 9)
                channel_pair.amplitude_offset_handling = AWGAmplitudeOffsetHandling.IGNORE_OFFSET
                channel_pair.upload('test2', program, (1, None), (None, None), test_transform)

                tabor_program_mock.assert_called_once_with(program, **tabor_program_kwargs,
                                                           sample_rate=test_sample_rate,
                                                           amplitudes=test_amplitudes,
                                                           offsets=(0., 0.),
                                                           voltage_transformations=test_transform)
                self.assertEqual([], offset_mock.call_args_list)

    def test_find_place_for_segments_in_memory(self):
        def hash_based_on_dir(ch):
            hash_list = []
            for d in dir(ch):
                o = getattr(ch, d)
                if isinstance(o, np.ndarray):
                    hash_list.append(hash(o.tobytes()))
                else:
                    try:
                        hash_list.append(hash(o))
                    except TypeError:
                        pass
            return hash(tuple(hash_list))

        channel_pair = TaborChannelPair(self.instrument, identifier='asd', channels=(1, 2))

        # empty
        segments = np.asarray([-5, -6, -7, -8, -9])
        segment_lengths = 192 + np.asarray([32, 16, 64, 32, 16])

        hash_before = hash_based_on_dir(channel_pair)

        w2s, ta, ti = channel_pair._find_place_for_segments_in_memory(segments, segment_lengths)
        self.assertEqual(w2s.tolist(), [-1, -1, -1, -1, -1])
        self.assertEqual(ta.tolist(), [True, True, True, True, True])
        self.assertEqual(ti.tolist(), [-1, -1, -1, -1, -1])
        self.assertEqual(hash_before, hash_based_on_dir(channel_pair))

        # all new segments
        channel_pair._segment_capacity = 192 + np.asarray([0, 16, 32, 16, 0], dtype=np.uint32)
        channel_pair._segment_hashes = np.asarray([1, 2, 3, 4, 5], dtype=np.int64)
        channel_pair._segment_references = np.asarray([1, 1, 1, 2, 1], dtype=np.int32)
        hash_before = hash_based_on_dir(channel_pair)

        w2s, ta, ti = channel_pair._find_place_for_segments_in_memory(segments, segment_lengths)
        self.assertEqual(w2s.tolist(), [-1, -1, -1, -1, -1])
        self.assertEqual(ta.tolist(), [True, True, True, True, True])
        self.assertEqual(ti.tolist(), [-1, -1, -1, -1, -1])
        self.assertEqual(hash_before, hash_based_on_dir(channel_pair))

        # some known segments
        channel_pair._segment_capacity = 192 + np.asarray([0, 16, 32, 64, 0, 16], dtype=np.uint32)
        channel_pair._segment_hashes = np.asarray([1, 2, 3, -7, 5, -9], dtype=np.int64)
        channel_pair._segment_references = np.asarray([1, 1, 1, 2, 1, 3], dtype=np.int32)
        hash_before = hash_based_on_dir(channel_pair)

        w2s, ta, ti = channel_pair._find_place_for_segments_in_memory(segments, segment_lengths)
        self.assertEqual(w2s.tolist(), [-1, -1, 3, -1, 5])
        self.assertEqual(ta.tolist(), [True, True, False, True, False])
        self.assertEqual(ti.tolist(), [-1, -1, -1, -1, -1])
        self.assertEqual(hash_before, hash_based_on_dir(channel_pair))

        # insert some segments with same length
        channel_pair._segment_capacity = 192 + np.asarray([0, 16, 32, 64, 0, 16], dtype=np.uint32)
        channel_pair._segment_hashes = np.asarray([1, 2, 3, 4, 5, 6], dtype=np.int64)
        channel_pair._segment_references = np.asarray([1, 0, 1, 0, 1, 3], dtype=np.int32)
        hash_before = hash_based_on_dir(channel_pair)

        w2s, ta, ti = channel_pair._find_place_for_segments_in_memory(segments, segment_lengths)
        self.assertEqual(w2s.tolist(), [-1, -1, -1, -1, -1])
        self.assertEqual(ta.tolist(), [True, False, False, True, True])
        self.assertEqual(ti.tolist(), [-1, 1, 3, -1, -1])
        self.assertEqual(hash_before, hash_based_on_dir(channel_pair))

        # insert some segments with smaller length
        channel_pair._segment_capacity = 192 + np.asarray([0, 80, 32, 64, 96, 16], dtype=np.uint32)
        channel_pair._segment_hashes = np.asarray([1, 2, 3, 4, 5, 6], dtype=np.int64)
        channel_pair._segment_references = np.asarray([1, 0, 1, 1, 0, 3], dtype=np.int32)
        hash_before = hash_based_on_dir(channel_pair)

        w2s, ta, ti = channel_pair._find_place_for_segments_in_memory(segments, segment_lengths)
        self.assertEqual(w2s.tolist(), [-1, -1, -1, -1, -1])
        self.assertEqual(ta.tolist(), [True, True, False, False, True])
        self.assertEqual(ti.tolist(), [-1, -1, 4, 1, -1])
        self.assertEqual(hash_before, hash_based_on_dir(channel_pair))

        # mix everything
        segments = np.asarray([-5, -6, -7, -8, -9, -10, -11])
        segment_lengths = 192 + np.asarray([32, 16, 64, 32, 16, 0, 0])

        channel_pair._segment_capacity = 192 + np.asarray([0, 80, 32, 64, 32, 16], dtype=np.uint32)
        channel_pair._segment_hashes = np.asarray([1, 2, 3, 4, -8, 6], dtype=np.int64)
        channel_pair._segment_references = np.asarray([1, 0, 1, 0, 1, 0], dtype=np.int32)
        hash_before = hash_based_on_dir(channel_pair)

        w2s, ta, ti = channel_pair._find_place_for_segments_in_memory(segments, segment_lengths)
        self.assertEqual(w2s.tolist(), [-1,    -1,   -1,    4,     -1,     -1, -1])
        self.assertEqual(ta.tolist(),  [False, True, False, False, True, True, True])
        self.assertEqual(ti.tolist(),  [1,     -1,   3,     -1,    -1,   -1,   -1])
        self.assertEqual(hash_before, hash_based_on_dir(channel_pair))

    def test_upload_segment(self):
        channel_pair = TaborChannelPair(self.instrument, identifier='asd', channels=(1, 2))

        self.reset_instrument_logs()

        channel_pair._segment_references = np.array([1, 2, 0, 1], dtype=np.uint32)
        channel_pair._segment_capacity = 192 + np.array([0, 16, 32, 32], dtype=np.uint32)
        channel_pair._segment_lengths = channel_pair._segment_capacity.copy()

        channel_pair._segment_hashes = np.array([1, 2, 3, 4], dtype=np.int64)

        # prevent entering and exiting configuration mode
        channel_pair._configuration_guard_count = 2

        segment = TaborSegment.from_sampled(np.ones(192+16, dtype=np.uint16), np.zeros(192+16, dtype=np.uint16), None, None)
        segment_binary = segment.get_as_binary()
        with self.assertRaises(ValueError):
            channel_pair._upload_segment(3, segment)

        with self.assertRaises(ValueError):
            channel_pair._upload_segment(0, segment)

        channel_pair._upload_segment(2, segment)
        np.testing.assert_equal(channel_pair._segment_capacity, 192 + np.array([0, 16, 32, 32], dtype=np.uint32))
        np.testing.assert_equal(channel_pair._segment_lengths, 192 + np.array([0, 16, 16, 32], dtype=np.uint32))
        np.testing.assert_equal(channel_pair._segment_hashes, np.array([1, 2, hash(segment), 4], dtype=np.int64))

        expected_commands = [':INST:SEL 1', ':INST:SEL 1', ':INST:SEL 1',
                             ':TRAC:DEF 3, 208',
                             ':TRAC:SEL 3',
                             ':TRAC:MODE COMB']
        expected_log = [((), dict(cmd_str=cmd, paranoia_level=channel_pair.internal_paranoia_level))
                        for cmd in expected_commands]
        self.assertAllCommandLogsEqual(expected_log)

        self.assert_written_segment_data(segment_binary)

    def test_amend_segments_flush(self):
        channel_pair = TaborChannelPair(self.instrument, identifier='asd', channels=(1, 2))
        # prevent entering and exiting configuration mode
        channel_pair._configuration_guard_count = 2

        self.instrument.main_instrument.paranoia_level = 0
        self.instrument.main_instrument.logged_commands = []
        self.instrument.main_instrument.logged_queries = []
        self.instrument.main_instrument._send_binary_data_calls = []
        self.reset_instrument_logs()

        channel_pair._segment_references = np.array([1, 2, 0, 1], dtype=np.uint32)
        channel_pair._segment_capacity = 192 + np.array([0, 16, 32, 32], dtype=np.uint32)
        channel_pair._segment_lengths = 192 + np.array([0, 16, 16, 32], dtype=np.uint32)

        channel_pair._segment_hashes = np.array([1, 2, 3, 4], dtype=np.int64)

        data = np.ones(192, dtype=np.uint16)
        segments = [TaborSegment.from_sampled(0*data, 1*data, None, None),
                    TaborSegment.from_sampled(1*data, 2*data, None, None)]

        channel_pair._amend_segments(segments)

        expected_references = np.array([1, 2, 0, 1, 1, 1], dtype=np.uint32)
        expected_capacities = 192 + np.array([0, 16, 32, 32, 0, 0], dtype=np.uint32)
        expected_lengths = 192 + np.array([0, 16, 16, 32, 0, 0], dtype=np.uint32)
        expected_hashes = np.array([1, 2, 3, 4, hash(segments[0]), hash(segments[1])], dtype=np.int64)

        np.testing.assert_equal(channel_pair._segment_references, expected_references)
        np.testing.assert_equal(channel_pair._segment_capacity, expected_capacities)
        np.testing.assert_equal(channel_pair._segment_lengths, expected_lengths)
        np.testing.assert_equal(channel_pair._segment_hashes, expected_hashes)

        expected_commands = [':INST:SEL 1',
                             ':TRAC:DEF 5,{}'.format(2 * 192 + 16),
                             ':TRAC:SEL 5',
                             ':TRAC:MODE COMB',
                             ':TRAC:DEF 3,208']
        expected_log = [((), dict(cmd_str=cmd, paranoia_level=channel_pair.internal_paranoia_level))
                        for cmd in expected_commands]
        self.assertAllCommandLogsEqual(expected_log)

        self.assert_written_segment_lengths(expected_capacities)

        expected_bin_blob = make_combined_wave(segments)

        self.assert_written_segment_data(expected_bin_blob)

    def test_amend_segments_iter(self):
        channel_pair = TaborChannelPair(self.instrument, identifier='asd', channels=(1, 2))
        # prevent entering and exiting configuration mode
        channel_pair._configuration_guard_count = 2

        self.instrument.paranoia_level = 0
        self.reset_instrument_logs()

        channel_pair._segment_references = np.array([1, 2, 0, 1], dtype=np.uint32)
        channel_pair._segment_capacity = 192 + np.array([0, 16, 32, 32], dtype=np.uint32)
        channel_pair._segment_lengths = 192 + np.array([0, 0, 16, 16], dtype=np.uint32)

        channel_pair._segment_hashes = np.array([1, 2, 3, 4], dtype=np.int64)

        data = np.ones(192, dtype=np.uint16)
        segments = [TaborSegment.from_sampled(0*data, 1*data, None, None),
                    TaborSegment.from_sampled(1*data, 2*data, None, None)]

        indices = channel_pair._amend_segments(segments)

        expected_references = np.array([1, 2, 0, 1, 1, 1], dtype=np.uint32)
        expected_capacities = 192 + np.array([0, 16, 32, 32, 0, 0], dtype=np.uint32)
        expected_lengths = 192 + np.array([0, 0, 16, 16, 0, 0], dtype=np.uint32)
        expected_hashes = np.array([1, 2, 3, 4, hash(segments[0]), hash(segments[1])], dtype=np.int64)

        np.testing.assert_equal(channel_pair._segment_references, expected_references)
        np.testing.assert_equal(channel_pair._segment_capacity, expected_capacities)
        np.testing.assert_equal(channel_pair._segment_lengths, expected_lengths)
        np.testing.assert_equal(channel_pair._segment_hashes, expected_hashes)

        np.testing.assert_equal(indices, np.array([4, 5], dtype=np.int64))

        expected_commands = [':INST:SEL 1',
                             ':TRAC:DEF 5,{}'.format(2 * 192 + 16),
                             ':TRAC:SEL 5',
                             ':TRAC:MODE COMB',
                             ':TRAC:DEF 5,192',
                             ':TRAC:DEF 6,192']
        expected_log = [((), dict(cmd_str=cmd, paranoia_level=channel_pair.internal_paranoia_level))
                        for cmd in expected_commands]
        self.assertAllCommandLogsEqual(expected_log)

        # no segment lengths written
        self.assert_written_segment_lengths()

        expected_bin_blob = make_combined_wave(segments)
        self.assert_written_segment_data(expected_bin_blob)

    def test_cleanup(self):
        channel_pair = TaborChannelPair(self.instrument, identifier='asd', channels=(1, 2))

        self.instrument.paranoia_level = 0
        self.reset_instrument_logs()

        channel_pair._segment_references = np.array([1, 2, 0, 1], dtype=np.uint32)
        channel_pair._segment_capacity = 192 + np.array([0, 16, 32, 32], dtype=np.uint32)
        channel_pair._segment_lengths = 192 + np.array([0, 0, 16, 16], dtype=np.uint32)
        channel_pair._segment_hashes = np.array([1, 2, 3, 4], dtype=np.int64)

        channel_pair.cleanup()
        np.testing.assert_equal(channel_pair._segment_references, np.array([1, 2, 0, 1], dtype=np.uint32))
        np.testing.assert_equal(channel_pair._segment_capacity, 192 + np.array([0, 16, 32, 32], dtype=np.uint32))
        np.testing.assert_equal(channel_pair._segment_lengths, 192 + np.array([0, 0, 16, 16], dtype=np.uint32))
        np.testing.assert_equal(channel_pair._segment_hashes, np.array([1, 2, 3, 4], dtype=np.int64))

        channel_pair._segment_references = np.array([1, 2, 0, 1, 0], dtype=np.uint32)
        channel_pair._segment_capacity = 192 + np.array([0, 16, 32, 32, 32], dtype=np.uint32)
        channel_pair._segment_lengths = 192 + np.array([0, 0, 16, 16, 0], dtype=np.uint32)
        channel_pair._segment_hashes = np.array([1, 2, 3, 4, 5], dtype=np.int64)

        channel_pair.cleanup()
        np.testing.assert_equal(channel_pair._segment_references, np.array([1, 2, 0, 1], dtype=np.uint32))
        np.testing.assert_equal(channel_pair._segment_capacity, 192 + np.array([0, 16, 32, 32], dtype=np.uint32))
        np.testing.assert_equal(channel_pair._segment_lengths, 192 + np.array([0, 0, 16, 16], dtype=np.uint32))
        np.testing.assert_equal(channel_pair._segment_hashes, np.array([1, 2, 3, 4], dtype=np.int64))

    def test_remove(self):
        channel_pair = TaborChannelPair(self.instrument, identifier='asd', channels=(1, 2))

        calls = []

        program_name = 'test'
        def dummy_free_program(name):
            self.assertIs(name, program_name)
            calls.append('free_program')

        def dummy_cleanup():
            calls.append('cleanup')

        channel_pair.cleanup = dummy_cleanup
        channel_pair.free_program = dummy_free_program

        channel_pair.remove(program_name)
        self.assertEqual(calls, ['free_program', 'cleanup'])

    def test_change_armed_program_single_sequence(self):
        channel_pair = TaborChannelPair(self.instrument, identifier='asd', channels=(1, 2))
        # prevent entering and exiting configuration mode
        channel_pair._configuration_guard_count = 2

        self.instrument.paranoia_level = 0
        self.instrument.logged_commands = []
        self.instrument.logged_queries = []
        self.reset_instrument_logs()

        advanced_sequencer_table = [(2, 1, 0)]
        sequencer_tables = [[(3, 0, 0), (2, 1, 0), (1, 0, 0), (1, 2, 0), (1, 3, 0)]]
        w2s = np.array([2, 5, 3, 1])

        sequencer_tables = self.to_new_sequencer_tables(sequencer_tables)
        advanced_sequencer_table = self.to_new_advanced_sequencer_table(advanced_sequencer_table)

        expected_sequencer_table = [(3, 3, 0), (2, 6, 0), (1, 3, 0), (1, 4, 0), (1, 2, 0)]

        program = DummyTaborProgramClass(advanced_sequencer_table=advanced_sequencer_table,
                                         sequencer_tables=sequencer_tables,
                                         waveform_mode=TaborSequencing.SINGLE)(None, None, None, None)

        channel_pair._known_programs['test'] = TaborProgramMemory(w2s, program)

        channel_pair.change_armed_program('test')

        expected_adv_seq_table_log = [([(1, 1, 1), (2, 2, 0), (1, 1, 0)], ':ASEQ:DATA', None)]
        expected_sequencer_table_log = [((sequencer_table,), dict(pref=':SEQ:DATA', paranoia_level=None))
                                        for sequencer_table in [channel_pair._idle_sequence_table,
                                                                expected_sequencer_table]]

        self.assert_written_sequencer_tables(channel_pair._idle_sequence_table, expected_sequencer_table)
        self.assert_written_advanced_sequencer_tables([(1, 1, 1), (2, 2, 0), (1, 1, 0)])

    def test_change_armed_program_single_waveform(self):
        channel_pair = TaborChannelPair(self.instrument, identifier='asd', channels=(1, 2))
        # prevent entering and exiting configuration mode
        channel_pair._configuration_guard_count = 2

        self.instrument.paranoia_level = 0
        self.instrument.logged_commands = []
        self.instrument.logged_queries = []
        self.reset_instrument_logs()

        advanced_sequencer_table = [(1, 1, 0)]
        sequencer_tables = [[(10, 0, 0)]]
        w2s = np.array([4])

        sequencer_tables = self.to_new_sequencer_tables(sequencer_tables)
        advanced_sequencer_table = self.to_new_advanced_sequencer_table(advanced_sequencer_table)

        expected_sequencer_table = [(10, 5, 0), (1, 1, 0), (1, 1, 0)]

        program = DummyTaborProgramClass(advanced_sequencer_table=advanced_sequencer_table,
                                         sequencer_tables=sequencer_tables,
                                         waveform_mode=TaborSequencing.SINGLE)(None, None, None, None)

        channel_pair._known_programs['test'] = TaborProgramMemory(w2s, program)

        channel_pair.change_armed_program('test')

        expected_adv_seq_table_log = [([(1, 1, 1), (1, 2, 0), (1, 1, 0)], ':ASEQ:DATA', None)]
        expected_sequencer_table_log = [((sequencer_table,), dict(pref=':SEQ:DATA', paranoia_level=None))
                                        for sequencer_table in [channel_pair._idle_sequence_table,
                                                                expected_sequencer_table]]

        self.assert_written_sequencer_tables(channel_pair._idle_sequence_table, expected_sequencer_table)
        self.assert_written_advanced_sequencer_tables([(1, 1, 1), (1, 2, 0), (1, 1, 0)])

    def test_change_armed_program_advanced_sequence(self):
        channel_pair = TaborChannelPair(self.instrument, identifier='asd', channels=(1, 2))
        # prevent entering and exiting configuration mode
        channel_pair._configuration_guard_count = 2

        self.instrument.paranoia_level = 0
        self.instrument.logged_commands = []
        self.instrument.logged_queries = []
        self.instrument._send_binary_data_calls = []

        self.reset_instrument_logs()

        advanced_sequencer_table = [(2, 1, 0), (3, 2, 0)]
        sequencer_tables = [[(3, 0, 0), (2, 1, 0), (1, 0, 0), (1, 2, 0), (1, 3, 0)],
                            [(4, 1, 0), (2, 1, 0), (1, 0, 0), (1, 2, 0), (1, 3, 0)]]
        wf_idx2seg_idx = np.array([2, 5, 3, 1])

        sequencer_tables = self.to_new_sequencer_tables(sequencer_tables)
        advanced_sequencer_table = self.to_new_advanced_sequencer_table(advanced_sequencer_table)

        expected_sequencer_tables = [[(3, 3, 0), (2, 6, 0), (1, 3, 0), (1, 4, 0), (1, 2, 0)],
                                     [(4, 6, 0), (2, 6, 0), (1, 3, 0), (1, 4, 0), (1, 2, 0)]]

        program = DummyTaborProgramClass(advanced_sequencer_table=advanced_sequencer_table,
                                         sequencer_tables=sequencer_tables,
                                         waveform_mode=TaborSequencing.ADVANCED)(None, None, None, None)

        channel_pair._known_programs['test'] = TaborProgramMemory(wf_idx2seg_idx, program)

        channel_pair.change_armed_program('test')

        expected_adv_seq_table_log = [([(1, 1, 1), (2, 2, 0), (3, 3, 0)], ':ASEQ:DATA', None)]
        expected_sequencer_table_log = [((sequencer_table,), dict(pref=':SEQ:DATA', paranoia_level=None))
                                        for sequencer_table in [channel_pair._idle_sequence_table] +
                                        expected_sequencer_tables]

        self.assert_written_sequencer_tables(channel_pair._idle_sequence_table, *expected_sequencer_tables)
        self.assert_written_advanced_sequencer_tables([(1, 1, 1), (2, 2, 0), (3, 3, 0)])
