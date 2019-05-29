from typing import cast
import unittest
import contextlib
from unittest import mock

import numpy as np

import qupulse.hardware.awgs.tektronix as tektronix
from qupulse.hardware.awgs.tektronix import TektronixAWG, TektronixProgram, parse_program, _make_binary_waveform
from qupulse._program._loop import Loop
from qupulse.utils.types import TimeType


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



        raise NotImplementedError()

    def test_init(self):
        raise NotImplementedError()


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
