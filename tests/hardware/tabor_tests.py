import unittest
import numbers
import itertools
from copy import copy, deepcopy
import numpy as np
import sys
from typing import List, Tuple, Dict

from qctoolkit.pulses.table_pulse_template import TableWaveform, HoldInterpolationStrategy
from qctoolkit.hardware.awgs.tabor import TaborAWGRepresentation, TaborException, TaborProgram, TaborChannelPair,\
    TaborSegment, TaborSequencing, TaborProgramMemory, with_configuration_guard
from qctoolkit.hardware.program import MultiChannelProgram, Loop
from qctoolkit.pulses.instructions import InstructionBlock
from qctoolkit.hardware.util import voltage_to_uint16, make_combined_wave

from teawg import model_properties_dict

from tests.pulses.sequencing_dummies import DummyWaveform
from tests.hardware import use_dummy_tabor
from tests.hardware.program_tests import LoopTests, WaveformGenerator, MultiChannelTests

hardware_instrument = None
def get_instrument():
    if use_dummy_tabor:
        instrument = TaborAWGRepresentation('main_instrument',
                                            reset=True,
                                            paranoia_level=2,
                                            mirror_addresses=['mirror_instrument'])
        instrument.main_instrument.visa_inst.answers[':OUTP:COUP'] = 'DC'
        instrument.main_instrument.visa_inst.answers[':VOLT'] = '1.0'
        instrument.main_instrument.visa_inst.answers[':FREQ:RAST'] = '1e9'
        instrument.main_instrument.visa_inst.answers[':VOLT:HV'] = '0.7'
        return instrument
    else:
        instrument_address = ('127.0.0.1', )
        if hardware_instrument is None:
            hardware_instrument = TaborAWGRepresentation(instrument_address,
                                            reset=True,
                                            paranoia_level=2)
            hardware_instrument.main_instrument.visa_inst.timeout = 25000

            if not hardware_instrument.is_open:
                raise RuntimeError('Could not connect to instrument')
        return hardware_instrument


def reset_instrument_logs(inst: TaborAWGRepresentation):
    for device in inst.all_devices:
        device.logged_commands = []
        device._send_binary_data_calls = []


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
    def assertAllCommandLogsEqual(self, inst: TaborAWGRepresentation, expected_log: List):
        for device in inst.all_devices:
            self.assertEqual(device.logged_commands, expected_log)

    def test_send_cmd(self):
        inst = get_instrument()

        reset_instrument_logs(inst)

        inst.send_cmd('bleh', paranoia_level=3)

        self.assertAllCommandLogsEqual(inst, [((), dict(paranoia_level=3, cmd_str='bleh'))])

        inst.send_cmd('bleho')
        self.assertAllCommandLogsEqual(inst, [((), dict(paranoia_level=3, cmd_str='bleh')),
                                              ((), dict(cmd_str='bleho', paranoia_level=None))])

    def test_trigger(self):
        inst = get_instrument()

        reset_instrument_logs(inst)
        inst.trigger()

        self.assertAllCommandLogsEqual(inst, [((), dict(cmd_str=':TRIG', paranoia_level=None))])

    def test_paranoia_level(self):
        inst = get_instrument()

        self.assertEqual(inst.paranoia_level, inst.main_instrument.paranoia_level)
        inst.paranoia_level = 30
        for device in inst.all_devices:
            self.assertEqual(device.paranoia_level, 30)

    def test_reset(self):
        inst = get_instrument()

        reset_instrument_logs(inst)

        inst.main_instrument.logged_commands = []
        inst.reset()

        expected_commands = [':RES',
                             ':INST:SEL 1; :INIT:GATE OFF; :INIT:CONT ON; '
                             ':INIT:CONT:ENAB ARM; :INIT:CONT:ENAB:SOUR BUS',
                             ':INST:SEL 3; :INIT:GATE OFF; :INIT:CONT ON; '
                             ':INIT:CONT:ENAB ARM; :INIT:CONT:ENAB:SOUR BUS']
        expected_log = [((), dict(cmd_str=cmd, paranoia_level=None))
                        for cmd in expected_commands]
        self.assertEqual(inst.main_instrument.logged_commands, expected_log)

    def test_enable(self):
        inst = get_instrument()

        reset_instrument_logs(inst)
        inst.enable()

        expected_commands = [':ENAB']
        expected_log = [((), dict(cmd_str=cmd, paranoia_level=None))
                        for cmd in expected_commands]
        self.assertAllCommandLogsEqual(inst, expected_log)


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

    def __call__(self, program: Loop, device_properties, channels, markers):
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


class TaborChannelPairTests(TaborAWGRepresentationDummyBasedTests):
    def test_copy(self):
        channel_pair = TaborChannelPair(get_instrument(), identifier='asd', channels=(1, 2))
        with self.assertRaises(NotImplementedError):
            copy(channel_pair)
        with self.assertRaises(NotImplementedError):
            deepcopy(channel_pair)

    def test_init(self):
        with self.assertRaises(ValueError):
            TaborChannelPair(get_instrument(), identifier='asd', channels=(1, 3))

    def test_free_program(self):
        instrument = get_instrument()
        channel_pair = TaborChannelPair(instrument, identifier='asd', channels=(1, 2))

        with self.assertRaises(KeyError):
            channel_pair.free_program('test')

        program = TaborProgramMemory(np.array([1, 2], dtype=np.int64), None)

        channel_pair._segment_references = np.array([1, 3, 1, 0])
        channel_pair._known_programs['test'] = program
        self.assertIs(channel_pair.free_program('test'), program)

        np.testing.assert_equal(channel_pair._segment_references, np.array([1, 2, 0, 0]))


    def test_upload_exceptions(self):
        wv = TableWaveform(1, [(0, 0.1, HoldInterpolationStrategy()),
                               (192, 0.1, HoldInterpolationStrategy())], [])

        channel_pair = TaborChannelPair(get_instrument(), identifier='asd', channels=(1, 2))

        program = Loop(waveform=wv)
        with self.assertRaises(ValueError):
            channel_pair.upload('test', program, (1, 2, 3), (5, 6), (lambda x: x, lambda x: x))
        with self.assertRaises(ValueError):
            channel_pair.upload('test', program, (1, 2), (5, 6, 'a'), (lambda x: x, lambda x: x))
        with self.assertRaises(ValueError):
            channel_pair.upload('test', program, (1, 2), (3, 4), (lambda x: x,))

        channel_pair._known_programs['test'] = TaborProgramMemory(np.array([0]), None)
        with self.assertRaises(ValueError):
            channel_pair.upload('test', program, (1, 2), (3, 4), (lambda x: x, lambda x: x))

    def test_upload(self):
        segments = np.array([1, 2, 3, 4, 5])
        segment_lengths = np.array([0, 16, 0, 16, 0], dtype=np.uint16)

        segment_references = np.array([1, 1, 2, 0, 1], dtype=np.uint32)

        w2s = np.array([-1, -1, 1, 2, -1], dtype=np.int64)
        ta = np.array([True, False, False, False, True])
        ti = np.array([-1, 3, -1, -1, -1])

        to_restore = sys.modules['qctoolkit.hardware.awgs.tabor'].TaborProgram
        my_class = DummyTaborProgramClass(segments=segments, segment_lengths=segment_lengths)
        sys.modules['qctoolkit.hardware.awgs.tabor'].TaborProgram = my_class
        try:
            program = Loop(waveform=DummyWaveform(duration=192))

            channel_pair = TaborChannelPair(get_instrument(), identifier='asd', channels=(1, 2))
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

            channel_pair._find_place_for_segments_in_memory = dummy_find_place
            channel_pair._upload_segment = dummy_upload_segment
            channel_pair._amend_segments = dummy_amend_segments

            channel_pair.upload('test', program, (1, None), (None, None), (lambda x: x, lambda x: x))

            self.assertIs(my_class.program, program)

            # the other references are increased in amend and upload segment method
            np.testing.assert_equal(channel_pair._segment_references, np.array([1, 2, 3, 0, 1]))

            self.assertEqual(len(channel_pair._known_programs), 1)
            np.testing.assert_equal(channel_pair._known_programs['test'].waveform_to_segment,
                                    np.array([5, 3, 1, 2, 6], dtype=np.int64))
            self.assertIs(channel_pair._known_programs['test'].program, my_class.created[0])

        finally:
            sys.modules['qctoolkit.hardware.awgs.tabor'].TaborProgram = to_restore

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

        channel_pair = TaborChannelPair(get_instrument(), identifier='asd', channels=(1, 2))

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
        instrument = get_instrument()
        channel_pair = TaborChannelPair(instrument, identifier='asd', channels=(1, 2))

        instrument.paranoia_level = 0
        reset_instrument_logs(instrument)

        channel_pair._segment_references = np.array([1, 2, 0, 1], dtype=np.uint32)
        channel_pair._segment_capacity = 192 + np.array([0, 16, 32, 32], dtype=np.uint32)
        channel_pair._segment_lengths = channel_pair._segment_capacity.copy()

        channel_pair._segment_hashes = np.array([1, 2, 3, 4], dtype=np.int64)

        # prevent entering and exiting configuration mode
        channel_pair._configuration_guard_count = 2

        segment = TaborSegment(np.ones(192+16, dtype=np.uint16), np.zeros(192+16, dtype=np.uint16))
        segment_binary = segment.get_as_binary()
        with self.assertRaises(ValueError):
            channel_pair._upload_segment(3, segment)

        with self.assertRaises(ValueError):
            channel_pair._upload_segment(0, segment)

        channel_pair._upload_segment(2, segment)
        np.testing.assert_equal(channel_pair._segment_capacity, 192 + np.array([0, 16, 32, 32], dtype=np.uint32))
        np.testing.assert_equal(channel_pair._segment_lengths, 192 + np.array([0, 16, 16, 32], dtype=np.uint32))
        np.testing.assert_equal(channel_pair._segment_hashes, np.array([1, 2, hash(segment), 4], dtype=np.int64))

        expected_commands = [':TRAC:DEF 3, 208',
                             ':TRAC:SEL 3',
                             ':TRAC:MODE COMB']
        expected_log = [((), dict(cmd_str=cmd, paranoia_level=None))
                        for cmd in expected_commands]
        self.assertAllCommandLogsEqual(instrument, expected_log)

        expected_send_binary_data_log = [(':TRAC:DATA', segment_binary, None)]
        for device in instrument.all_devices:
            np.testing.assert_equal(device._send_binary_data_calls, expected_send_binary_data_log)

    def test_amend_segments_flush(self):
        instrument = get_instrument()
        channel_pair = TaborChannelPair(instrument, identifier='asd', channels=(1, 2))
        # prevent entering and exiting configuration mode
        channel_pair._configuration_guard_count = 2

        instrument.main_instrument.paranoia_level = 0
        instrument.main_instrument.logged_commands = []
        instrument.main_instrument.logged_queries = []
        instrument.main_instrument._send_binary_data_calls = []

        channel_pair._segment_references = np.array([1, 2, 0, 1], dtype=np.uint32)
        channel_pair._segment_capacity = 192 + np.array([0, 16, 32, 32], dtype=np.uint32)
        channel_pair._segment_lengths = 192 + np.array([0, 16, 16, 32], dtype=np.uint32)

        channel_pair._segment_hashes = np.array([1, 2, 3, 4], dtype=np.int64)

        data = np.ones(192, dtype=np.uint16)
        segments = [TaborSegment(0*data, 1*data),
                    TaborSegment(1*data, 2*data)]

        channel_pair._amend_segments(segments)

        expected_references = np.array([1, 2, 0, 1, 1, 1], dtype=np.uint32)
        expected_capacities = 192 + np.array([0, 16, 32, 32, 0, 0], dtype=np.uint32)
        expected_lengths = 192 + np.array([0, 16, 16, 32, 0, 0], dtype=np.uint32)
        expected_hashes = np.array([1, 2, 3, 4, hash(segments[0]), hash(segments[1])], dtype=np.int64)

        np.testing.assert_equal(channel_pair._segment_references, expected_references)
        np.testing.assert_equal(channel_pair._segment_capacity, expected_capacities)
        np.testing.assert_equal(channel_pair._segment_lengths, expected_lengths)
        np.testing.assert_equal(channel_pair._segment_hashes, expected_hashes)

        expected_commands = [':TRAC:DEF 5,{}'.format(2 * 192 + 16),
                             ':TRAC:SEL 5',
                             ':TRAC:MODE COMB',
                             ':TRAC:DEF 3,208']
        expected_log = [((), dict(cmd_str=cmd, paranoia_level=None))
                        for cmd in expected_commands]
        self.assertEqual(expected_log, instrument.main_instrument.logged_commands)

        expected_download_segment_calls = [(expected_capacities, ':SEGM:DATA', None)]
        np.testing.assert_equal(instrument.main_instrument._download_segment_lengths_calls, expected_download_segment_calls)

        expected_bin_blob = make_combined_wave(segments)
        expected_send_binary_data_log = [(':TRAC:DATA', expected_bin_blob, None)]
        np.testing.assert_equal(instrument.main_instrument._send_binary_data_calls, expected_send_binary_data_log)

    def test_amend_segments_iter(self):
        instrument = get_instrument()

        channel_pair = TaborChannelPair(instrument, identifier='asd', channels=(1, 2))
        # prevent entering and exiting configuration mode
        channel_pair._configuration_guard_count = 2

        instrument.paranoia_level = 0
        reset_instrument_logs(instrument)

        channel_pair._segment_references = np.array([1, 2, 0, 1], dtype=np.uint32)
        channel_pair._segment_capacity = 192 + np.array([0, 16, 32, 32], dtype=np.uint32)
        channel_pair._segment_lengths = 192 + np.array([0, 0, 16, 16], dtype=np.uint32)

        channel_pair._segment_hashes = np.array([1, 2, 3, 4], dtype=np.int64)

        data = np.ones(192, dtype=np.uint16)
        segments = [TaborSegment(0*data, 1*data),
                    TaborSegment(1*data, 2*data)]

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

        expected_commands = [':TRAC:DEF 5,{}'.format(2 * 192 + 16),
                             ':TRAC:SEL 5',
                             ':TRAC:MODE COMB',
                             ':TRAC:DEF 5,192',
                             ':TRAC:DEF 6,192']
        expected_log = [((), dict(cmd_str=cmd, paranoia_level=None))
                        for cmd in expected_commands]
        self.assertAllCommandLogsEqual(instrument, expected_log)

        expected_download_segment_calls = []
        for device in instrument.all_devices:
            self.assertEqual(device._download_segment_lengths_calls, expected_download_segment_calls)

        expected_bin_blob = make_combined_wave(segments)
        expected_send_binary_data_log = [(':TRAC:DATA', expected_bin_blob, None)]
        for device in instrument.all_devices:
            np.testing.assert_equal(device._send_binary_data_calls, expected_send_binary_data_log)

    def test_cleanup(self):
        instrument = get_instrument()
        channel_pair = TaborChannelPair(instrument, identifier='asd', channels=(1, 2))

        instrument.paranoia_level = 0
        instrument.logged_commands = []
        instrument.logged_queries = []
        instrument._send_binary_data_calls = []

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
        instrument = get_instrument()
        channel_pair = TaborChannelPair(instrument, identifier='asd', channels=(1, 2))

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
        instrument = get_instrument()
        channel_pair = TaborChannelPair(instrument, identifier='asd', channels=(1, 2))
        # prevent entering and exiting configuration mode
        channel_pair._configuration_guard_count = 2

        instrument.paranoia_level = 0
        instrument.logged_commands = []
        instrument.logged_queries = []
        instrument._send_binary_data_calls = []

        advanced_sequencer_table = [(2, 1, 0)]
        sequencer_tables = [[(3, 0, 0), (2, 1, 0), (1, 0, 0), (1, 2, 0), (1, 3, 0)]]
        w2s = np.array([2, 5, 3, 1])

        expected_sequencer_table = [(3, 3, 0), (2, 6, 0), (1, 3, 0), (1, 4, 0), (1, 2, 0)]
        idle_sequencer_table = [(1, 1, 0), (1, 1, 0), (1, 1, 0)]

        program = DummyTaborProgramClass(advanced_sequencer_table=advanced_sequencer_table,
                                         sequencer_tables=sequencer_tables,
                                         waveform_mode=TaborSequencing.SINGLE)(None, None, None, None)

        channel_pair._known_programs['test'] = TaborProgramMemory(w2s, program)

        channel_pair.change_armed_program('test')

        expected_adv_seq_table_log = [([(1, 1, 1), (2, 2, 0), (1, 1, 0)], ':ASEQ:DATA', None)]
        expected_sequencer_table_log = [((sequencer_table,), dict(pref=':SEQ:DATA', paranoia_level=None))
                                        for sequencer_table in [idle_sequencer_table, expected_sequencer_table]]

        for device in instrument.all_devices:
            self.assertEqual(device._download_adv_seq_table_calls, expected_adv_seq_table_log)
            self.assertEqual(device._download_sequencer_table_calls, expected_sequencer_table_log)

    def test_change_armed_program_single_waveform(self):
        instrument = get_instrument()
        channel_pair = TaborChannelPair(instrument, identifier='asd', channels=(1, 2))
        # prevent entering and exiting configuration mode
        channel_pair._configuration_guard_count = 2

        instrument.paranoia_level = 0
        instrument.logged_commands = []
        instrument.logged_queries = []
        instrument._send_binary_data_calls = []

        advanced_sequencer_table = [(1, 1, 0)]
        sequencer_tables = [[(10, 0, 0)]]
        w2s = np.array([4])

        expected_sequencer_table = [(10, 5, 0), (1, 1, 0), (1, 1, 0)]
        idle_sequencer_table = [(1, 1, 0), (1, 1, 0), (1, 1, 0)]

        program = DummyTaborProgramClass(advanced_sequencer_table=advanced_sequencer_table,
                                         sequencer_tables=sequencer_tables,
                                         waveform_mode=TaborSequencing.SINGLE)(None, None, None, None)

        channel_pair._known_programs['test'] = TaborProgramMemory(w2s, program)

        channel_pair.change_armed_program('test')

        expected_adv_seq_table_log = [([(1, 1, 1), (1, 2, 0), (1, 1, 0)], ':ASEQ:DATA', None)]
        expected_sequencer_table_log = [((sequencer_table,), dict(pref=':SEQ:DATA', paranoia_level=None))
                                        for sequencer_table in [idle_sequencer_table, expected_sequencer_table]]

        for device in instrument.all_devices:
            self.assertEqual(device._download_adv_seq_table_calls, expected_adv_seq_table_log)
            self.assertEqual(device._download_sequencer_table_calls, expected_sequencer_table_log)

    def test_change_armed_program_advanced_sequence(self):
        instrument = get_instrument()
        channel_pair = TaborChannelPair(instrument, identifier='asd', channels=(1, 2))
        # prevent entering and exiting configuration mode
        channel_pair._configuration_guard_count = 2

        instrument.paranoia_level = 0
        instrument.logged_commands = []
        instrument.logged_queries = []
        instrument._send_binary_data_calls = []

        advanced_sequencer_table = [(2, 1, 0), (3, 2, 0)]
        sequencer_tables = [[(3, 0, 0), (2, 1, 0), (1, 0, 0), (1, 2, 0), (1, 3, 0)],
                            [(4, 1, 0), (2, 1, 0), (1, 0, 0), (1, 2, 0), (1, 3, 0)]]
        wf_idx2seg_idx = np.array([2, 5, 3, 1])

        idle_sequencer_table = [(1, 1, 0), (1, 1, 0), (1, 1, 0)]
        expected_sequencer_tables = [idle_sequencer_table,
                                     [(3, 3, 0), (2, 6, 0), (1, 3, 0), (1, 4, 0), (1, 2, 0)],
                                     [(4, 6, 0), (2, 6, 0), (1, 3, 0), (1, 4, 0), (1, 2, 0)]]

        program = DummyTaborProgramClass(advanced_sequencer_table=advanced_sequencer_table,
                                         sequencer_tables=sequencer_tables,
                                         waveform_mode=TaborSequencing.ADVANCED)(None, None, None, None)

        channel_pair._known_programs['test'] = TaborProgramMemory(wf_idx2seg_idx, program)

        channel_pair.change_armed_program('test')

        expected_adv_seq_table_log = [([(1, 1, 1), (2, 2, 0), (3, 3, 0)], ':ASEQ:DATA', None)]
        expected_sequencer_table_log = [((sequencer_table,), dict(pref=':SEQ:DATA', paranoia_level=None))
                                        for sequencer_table in expected_sequencer_tables]

        for device in instrument.all_devices:
            self.assertEqual(device._download_adv_seq_table_calls, expected_adv_seq_table_log)
            self.assertEqual(device._download_sequencer_table_calls, expected_sequencer_table_log)