from typing import cast
import unittest
import contextlib
from unittest import mock

import qctoolkit.hardware.awgs.tektronix as tektronix
from qctoolkit.hardware.awgs.tektronix import TektronixAWG, TektronixProgram, parse_program


class TektronixProgramTests(unittest.TestCase):
    def test_parse_program(self):
        raise NotImplementedError()

    def test_init(self):
        raise NotImplementedError()


class DummyTekAwg:
    def __init__(self, **kwargs):
        pass


class TektronixAWGTests(unittest.TestCase):
    def make_dummy_tek_awg(self, **kwargs) -> tektronix.TekAwg.TekAwg:
        if tektronix.TekAwg:
            return cast(tektronix.TekAwg.TekAwg, DummyTekAwg(**kwargs))

    def make_awg(self, **kwargs):
        make_waveform_patch = mock.patch('qctoolkit.hardware.awgs.tektronix.TektronixAWG.make_idle_waveform')
        clear_patch = mock.patch('qctoolkit.hardware.awgs.tektronix.TektronixAWG.clear')
        init_idle_patch = mock.patch('qctoolkit.hardware.awgs.tektronix.TektronixAWG.initialize_idle_program')
        synchronize_patch = mock.patch('qctoolkit.hardware.awgs.tektronix.TektronixAWG.synchronize')

        kwargs.setdefault('tek_awg', self.make_dummy_tek_awg())
        kwargs.setdefault('synchronize', 'read')

        with make_waveform_patch, clear_patch, init_idle_patch, synchronize_patch:
            return TektronixAWG(**kwargs)

    def test_init(self):
        make_waveform_patch = mock.patch('qctoolkit.hardware.awgs.tektronix.TektronixAWG.make_idle_waveform')
        clear_patch = mock.patch('qctoolkit.hardware.awgs.tektronix.TektronixAWG.clear')
        init_idle_patch = mock.patch('qctoolkit.hardware.awgs.tektronix.TektronixAWG.initialize_idle_program')
        synchronize_patch = mock.patch('qctoolkit.hardware.awgs.tektronix.TektronixAWG.synchronize')

        with mock.patch('qctoolkit.hardware.awgs.tektronix.TekAwg', new=None):
            with self.assertRaisesRegex(RuntimeError, 'TekAwg'):
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
