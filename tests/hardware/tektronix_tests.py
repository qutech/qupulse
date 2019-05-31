from typing import cast
import unittest
import contextlib
from unittest import mock

import numpy as np

import qupulse.hardware.awgs.tektronix as tektronix
from qupulse.hardware.awgs.tektronix import TektronixAWG, TektronixProgram, parse_program, _make_binary_waveform
from qupulse._program._loop import Loop
from qupulse.utils.types import TimeType
from tests.pulses.sequencing_dummies import DummyWaveform
from qupulse._program.waveforms import MultiChannelWaveform

class TektronixProgramTests(unittest.TestCase):
    @mock.patch('qupulse.hardware.awgs.tektronix.voltage_to_uint16')
    @mock.patch('tek_awg.Waveform')
    def test_make_binary_waveform(self, TekWf, mock_volt_to_bin):
        def get_sampled(channel, sample_times):
            return channel

        transformed = [4, 5, 6]
        def trafo(in_arr):
            return transformed

        tek_wf = mock.Mock()
        TekWf.return_value = tek_wf

        bin_data = [7, 8, 9]
        mock_volt_to_bin.return_value = bin_data

        waveform = mock.MagicMock()
        waveform.get_sampled = mock.Mock(wraps=get_sampled)

        mock_trafo = mock.Mock(wraps=trafo)
        voltage_to_uint16_kwargs = dict(asd='foo', f='bar')

        time_array = [1, 2, 3]
        channels = ('A', 'B', 'C')

        result = _make_binary_waveform(waveform, time_array, *channels, mock_trafo, voltage_to_uint16_kwargs)

        waveform.get_sampled.assert_any_call(channel='A', sample_times=time_array)
        waveform.get_sampled.assert_any_call(channel='B', sample_times=time_array)
        waveform.get_sampled.assert_any_call(channel='C', sample_times=time_array)

        mock_trafo.assert_called_once_with('A')

        mock_volt_to_bin.assert_called_once_with(transformed, **voltage_to_uint16_kwargs)
        TekWf.assert_called_once_with(channel=bin_data, marker_1='B', marker_2='C')

        self.assertIs(result, tek_wf)

    def test_parse_program(self):
        ill_formed_program = Loop(children=[Loop(children=[Loop()])])

        with self.assertRaisesRegex(AssertionError, 'Invalid program depth'):
            parse_program(ill_formed_program, (), (), TimeType(), (), (), ())

        channels = ('A', 'B', None, None)
        markers = (('A1', None), (None, None), (None, 'C2'), (None, None))

        # we do test offset handling separately
        amplitudes = (1, 1, 1, 1)
        offsets = (0, 0, 0, 0)
        voltage_transformations = tuple(mock.Mock(wraps=lambda x: x) for _ in range(4))

        used_channels = {'A', 'B', 'A1', 'C2'}
        wf_defined_channels = used_channels & {'other'}

        sampled_6 = [np.zeros((6,)),
                     np.arange(6) / 6,
                     np.ones((6,)) * 0.42,
                     np.array([0., .1, .2, 0., 1, 0]),
                     np.array([1., .0, .0, 0., 0, 1])]

        sampled_4 = [np.zeros((4,)),
                     np.arange(-4, 0) / 4,
                     np.ones((4,)) * 0.2,
                     np.array([0., -.1, -.2, 0.]),
                     np.array([0., 0, 0, 1.])]

        sample_rate_in_GHz = TimeType(1, 2)

        # channel A is the same in wfs_6[1] and wfs_6[2]
        wfs_6 = [DummyWaveform(duration=12, sample_output={'A':  sampled_6[0],
                                                           'B':  sampled_6[0],
                                                           'A1': sampled_6[0],
                                                           'C2': sampled_6[0]}, defined_channels=used_channels),
                 DummyWaveform(duration=12, sample_output={'A': sampled_6[1],
                                                           'B': sampled_6[2],
                                                           'A1': sampled_6[3],
                                                           'C2': sampled_6[4]}, defined_channels=used_channels),
                 DummyWaveform(duration=12, sample_output={'A': sampled_6[1],
                                                           'B': sampled_6[0],
                                                           'A1': sampled_6[3],
                                                           'C2': sampled_6[2]}, defined_channels=used_channels)
                 ]
        wfs_4 = [DummyWaveform(duration=8, sample_output={'A': sampled_4[0],
                                                           'B':sampled_4[0],
                                                           'A1': sampled_4[2],
                                                           'C2': sampled_4[3]}, defined_channels=used_channels),
                 DummyWaveform(duration=8, sample_output={'A': sampled_4[1],
                                                          'B': sampled_4[2],
                                                          'A1': sampled_4[2],
                                                          'C2': sampled_4[3]}, defined_channels=used_channels),
                 DummyWaveform(duration=8, sample_output={'A': sampled_4[2],
                                                          'B': sampled_4[0],
                                                          'A1': sampled_4[2],
                                                          'C2': sampled_4[3]}, defined_channels=used_channels)
                 ]

        # unset is equal to sampled_n[0]
        binary_waveforms_6 = [(0, 0, 0), (0, 0, 0), (0, 0, 0),
                              (1, 3, 0), (2, 0, 0), (0, 0, 4),
                              (1, 3, 0), (0, 0, 0), (0, 0, 2)]

        binary_waveforms_4 = [(0, 2, 0), (0, 0, 0), (0, 0, 3),
                              (1, 2, 0), (2, 0, 0), (0, 0, 3),
                              (2, 2, 0), (0, 0, 0), (0, 0, 3)]

        n_bin_waveforms = len(set(binary_waveforms_6)) + len(set(binary_waveforms_4))

        program = [(wfs_6[0], 1),
                   (wfs_4[0], 2),
                   (wfs_6[0], 3),
                   (wfs_6[1], 4),
                   (wfs_4[1], 5),
                   (wfs_6[2], 6),
                   (wfs_4[2], 7),
                   (wfs_6[1], 8),
                   (wfs_6[2], 9)]

        loop_program = Loop(children=[
            Loop(waveform=waveform, repetition_count=repetition_count)
            for waveform, repetition_count in program
        ])

        sequence_entries, waveforms = parse_program(
            program=loop_program,
            channels=channels,
            markers=markers,
            sample_rate=sample_rate_in_GHz * 10**9,
            amplitudes=amplitudes,
            voltage_transformations=voltage_transformations,
            offsets=offsets
        )

        waveform_set = set(waveforms)
        self.assertEqual(len(waveform_set), len(waveforms))
        self.assertIn(4, waveforms)
        self.assertIn(6, waveforms)

        waveform_set = waveform_set - {4, 6}
        self.assertEqual(len(waveform_set), n_bin_waveforms)

        self.assertEqual(len(sequence_entries), 9)

        raise NotImplementedError("Validate binary waveforms")

    @mock.patch('qupulse.hardware.awgs.tektronix.make_compatible')
    @mock.patch('qupulse.hardware.awgs.tektronix.parse_program')
    def test_init(self, mock_parse_program, mock_make_compatible):
        mock_parse_program.return_value = ('seq_el', 'wfs')

        mock_program = mock.MagicMock(spec=Loop)
        copied = mock.MagicMock(spec=Loop)
        channels = ('A', 'B', None, None)
        markers = (('A1', None), (None, None), (None, 'C2'), (None, None))
        sample_rate = TimeType(12)
        amplitudes = (1, 1, 1, 1)
        offsets = (0, 0, 0, 0)
        voltage_transformations = tuple(mock.Mock(wraps=lambda x: x) for _ in range(4))

        mock_program.copy_tree_structure.return_value = copied

        tek_program = TektronixProgram(program=mock_program,
                                       channels=channels,
                                       markers=markers,
                                       sample_rate=sample_rate,
                                       amplitudes=amplitudes,
                                       offsets=offsets,
                                       voltage_transformations=voltage_transformations)

        self.assertIs(tek_program._program, copied)
        copied.flatten_and_balance.assert_called_once_with(1)
        mock_make_compatible.assert_called_once_with(copied, 250, 1, sample_rate / 10**9)
        mock_parse_program.assert_called_once_with(program=copied,
                                       channels=channels,
                                       markers=markers,
                                       sample_rate=sample_rate,
                                       amplitudes=amplitudes,
                                       offsets=offsets,
                                       voltage_transformations=voltage_transformations)


class DummyTekAwg:
    def __init__(self, **kwargs):
        pass

    def write(self):
        raise NotImplementedError()


class TektronixAWGTests(unittest.TestCase):
    def make_dummy_tek_awg(self, **kwargs) -> tektronix.tek_awg.TekAwg:
        if tektronix.tek_awg:
            return cast(tektronix.tek_awg.TekAwg, DummyTekAwg(**kwargs))

    def make_awg(self, **kwargs):
        make_waveform_patch = mock.patch('qupulse.hardware.awgs.tektronix.TektronixAWG.make_idle_waveform')
        clear_patch = mock.patch('qupulse.hardware.awgs.tektronix.TektronixAWG.clear')
        init_idle_patch = mock.patch('qupulse.hardware.awgs.tektronix.TektronixAWG.initialize_idle_program')
        synchronize_patch = mock.patch('qupulse.hardware.awgs.tektronix.TektronixAWG.synchronize')

        kwargs.setdefault('tek_awg', self.make_dummy_tek_awg())
        kwargs.setdefault('synchronize', 'read')

        with make_waveform_patch, clear_patch, init_idle_patch, synchronize_patch:
            return TektronixAWG(**kwargs)

    def test_init(self):
        make_waveform_patch = mock.patch('qupulse.hardware.awgs.tektronix.TektronixAWG.make_idle_waveform')
        clear_patch = mock.patch('qupulse.hardware.awgs.tektronix.TektronixAWG.clear')
        init_idle_patch = mock.patch('qupulse.hardware.awgs.tektronix.TektronixAWG.initialize_idle_program')
        synchronize_patch = mock.patch('qupulse.hardware.awgs.tektronix.TektronixAWG.synchronize')

        with mock.patch('qupulse.hardware.awgs.tektronix.tek_awg', new=None):
            with self.assertRaisesRegex(RuntimeError, 'tek_awg'):
                TektronixAWG(self.make_dummy_tek_awg(), 'clear')

        with make_waveform_patch as make_idle_waveform:
            with self.assertRaisesRegex(ValueError, 'synchronize'):
                TektronixAWG(self.make_dummy_tek_awg(), 'foo')

            make_idle_waveform.assert_called_once_with(4000)

        with make_waveform_patch as make_idle_waveform, clear_patch as clear, init_idle_patch as init_idle:
            TektronixAWG(self.make_dummy_tek_awg(), 'clear')
            make_idle_waveform.assert_called_once_with(4000)
            clear.assert_called_once_with()
            init_idle.assert_called_once_with()

        with make_waveform_patch as make_idle_waveform, synchronize_patch as synchronize, init_idle_patch as init_idle:
            dummy = self.make_dummy_tek_awg()
            tek_awg = TektronixAWG(dummy, 'read')
            make_idle_waveform.assert_called_once_with(4000)
            synchronize.assert_called_once_with()
            init_idle.assert_called_once_with()

            self.assertIs(tek_awg.device, dummy)
            self.assertIsNone(tek_awg._cleanup_stack)

        tek_awg = self.make_awg(manual_cleanup=True)
        self.assertIsInstance(tek_awg._cleanup_stack, contextlib.ExitStack)

    def test_clear_waveforms(self):
        tek_awg = self.make_awg()

        with mock.patch.object(tek_awg.device, 'write') as dev_write, \
                mock.patch.object(tek_awg, 'read_waveforms') as read_waveforms:
            tek_awg._clear_waveforms()

            dev_write.assert_called_once_with('WLIS:WAV:DEL ALL')
            read_waveforms.assert_called_once_with()

    def test_clear_sequence(self):
        tek_awg = self.make_awg()

        with mock.patch.object(tek_awg.device, 'write') as dev_write, \
                mock.patch.object(tek_awg, 'read_sequence') as read_sequence:
            tek_awg._clear_sequence()

            dev_write.assert_called_once_with('SEQ:LENG 0')
            read_sequence.assert_called_once_with()
