import unittest
from unittest import mock
from collections import OrderedDict

import numpy as np

from qupulse.utils.types import TimeType
from qupulse.program.loop import Loop
from qupulse.program.waveforms import FunctionWaveform
from qupulse.expressions import Expression, ExpressionScalar
from qupulse.hardware.awgs.base import ProgramEntry

from tests.pulses.sequencing_dummies import DummyWaveform


class ProgramEntryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.channels = ('A', None, 'C')
        self.marker = (None, 'M')
        self.amplitudes = (1., 1., .5)
        self.offset = (0., .5, .1)
        self.voltage_transformations = (
            mock.Mock(wraps=lambda x: x),
            mock.Mock(wraps=lambda x: x),
            mock.Mock(wraps=lambda x: x)
        )
        self.sample_rate = TimeType.from_float(1)

        N = 100
        t = np.arange(N)

        self.sampled = [
            dict(A=np.linspace(-.1, .1, num=N), C=.1*np.sin(t), M=np.arange(N) % 2),
            dict(A=np.linspace(.1, -.1, num=N//2), C=.1 * np.cos(t[::2]), M=np.arange(N//2) % 3)
        ]
        self.waveforms = [
            wf
            for wf in (DummyWaveform(sample_output=sampled, duration=sampled['A'].size) for sampled in self.sampled)
        ]
        self.loop = Loop(children=[Loop(waveform=wf) for wf in self.waveforms] * 2)

    def test_init(self):
        sampled = [mock.Mock(), mock.Mock()]
        expected_default = OrderedDict([(wf, None) for wf in self.waveforms]).keys()
        expected_waveforms = OrderedDict(zip(self.waveforms, sampled))

        with mock.patch.object(ProgramEntry, '_sample_waveforms', return_value=sampled) as sample_waveforms:
            entry = ProgramEntry(program=self.loop,
                                 channels=self.channels,
                                 markers=self.marker,
                                 amplitudes=self.amplitudes,
                                 offsets=self.offset,
                                 voltage_transformations=self.voltage_transformations,
                                 sample_rate=self.sample_rate,
                                 waveforms=[],
                                )
            self.assertIs(self.loop, entry._loop)
            self.assertEqual(0, len(entry._waveforms))
            sample_waveforms.assert_not_called()

        with mock.patch.object(ProgramEntry, '_sample_waveforms', return_value=sampled) as sample_waveforms:
            entry = ProgramEntry(program=self.loop,
                                 channels=self.channels,
                                 markers=self.marker,
                                 amplitudes=self.amplitudes,
                                 offsets=self.offset,
                                 voltage_transformations=self.voltage_transformations,
                                 sample_rate=self.sample_rate,
                                 waveforms=None,
                                 )
            self.assertEqual(expected_waveforms, entry._waveforms)
            sample_waveforms.assert_called_once_with(expected_default)

        with mock.patch.object(ProgramEntry, '_sample_waveforms', return_value=sampled[:1]) as sample_waveforms:
            entry = ProgramEntry(program=self.loop,
                                 channels=self.channels,
                                 markers=self.marker,
                                 amplitudes=self.amplitudes,
                                 offsets=self.offset,
                                 voltage_transformations=self.voltage_transformations,
                                 sample_rate=self.sample_rate,
                                 waveforms=self.waveforms[:1],
                                 )
            self.assertEqual(OrderedDict([(self.waveforms[0], sampled[0])]), entry._waveforms)
            sample_waveforms.assert_called_once_with(self.waveforms[:1])

    def test_sample_waveforms(self):
        empty_ch = np.array([1, 2, 3])
        empty_m = np.array([0, 1, 0])
        # channels ==  (A, None, C)

        expected_sampled = [
            ((self.sampled[0]['A'], empty_ch, 2.*(self.sampled[0]['C'] - 0.1)), (empty_m, self.sampled[0]['M'] != 0)),
            ((self.sampled[1]['A'], empty_ch, 2.*(self.sampled[1]['C'] - 0.1)), (empty_m, self.sampled[1]['M'] != 0))
        ]

        entry = ProgramEntry(program=self.loop,
                             channels=self.channels,
                             markers=self.marker,
                             amplitudes=self.amplitudes,
                             offsets=self.offset,
                             voltage_transformations=self.voltage_transformations,
                             sample_rate=self.sample_rate,
                             waveforms=[],
                             )

        with mock.patch.object(entry, '_sample_empty_channel', return_value=empty_ch):
            with mock.patch.object(entry, '_sample_empty_marker', return_value=empty_m):
                sampled = entry._sample_waveforms(self.waveforms)
                np.testing.assert_equal(expected_sampled, sampled)


class ProgramEntryDivisorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.channels = ('A',)
        self.marker = tuple()
        self.amplitudes = (1.,)
        self.offset = (0.,)
        self.voltage_transformations = (
            mock.Mock(wraps=lambda x: x),
        )
        self.sample_rate = TimeType.from_float(2.4)

        t = np.arange(0,400/12,1/2.4)

        self.sampled = [
            dict(A=np.sin(t)),
            dict(A=np.sin(t[::8])),
        ]
        
        wf = FunctionWaveform(ExpressionScalar('sin(t)'), 400/12, 'A')
        wf2 = FunctionWaveform(ExpressionScalar('sin(t)'), 400/12, 'A')
        wf2._pow_2_divisor = 3
        self.waveforms = [
            wf,wf2
        ]
        self.loop = Loop(children=[Loop(waveform=wf) for wf in self.waveforms])

    def test_sample_waveforms_with_divisor(self):
        empty_ch = np.array([1,])
        empty_m = np.array([])
        # channels ==  (A,)

        expected_sampled = [
            ((self.sampled[0]['A'],), tuple()),
            ((self.sampled[1]['A'],), tuple()),
        ]

        entry = ProgramEntry(program=self.loop,
                             channels=self.channels,
                             markers=self.marker,
                             amplitudes=self.amplitudes,
                             offsets=self.offset,
                             voltage_transformations=self.voltage_transformations,
                             sample_rate=self.sample_rate,
                             waveforms=[])

        with mock.patch.object(entry, '_sample_empty_channel', return_value=empty_ch):
            with mock.patch.object(entry, '_sample_empty_marker', return_value=empty_m):
                sampled = entry._sample_waveforms(self.waveforms)
                np.testing.assert_equal(expected_sampled, sampled)
