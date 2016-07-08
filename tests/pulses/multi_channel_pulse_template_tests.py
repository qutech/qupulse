import unittest
import copy

import numpy

from qctoolkit.expressions import Expression
from qctoolkit.pulses.table_pulse_template import TablePulseTemplate
from qctoolkit.pulses.sequence_pulse_template import SequencePulseTemplate, MissingMappingException, UnnecessaryMappingException, MissingParameterDeclarationException
from qctoolkit.pulses.parameters import ParameterDeclaration, ParameterNotProvidedException, ConstantParameter
from qctoolkit.pulses.multi_channel_pulse_template import MultiChannelPulseTemplate, MultiChannelWaveform

from tests.pulses.sequencing_dummies import DummySequencer, DummyInstructionBlock, DummyPulseTemplate, DummyParameter, DummyNoValueParameter, DummyWaveform
from tests.serialization_dummies import DummySerializer


class MultiChannelWaveformTest(unittest.TestCase):

    def test_init_no_args(self) -> None:
        with self.assertRaises(ValueError):
            MultiChannelWaveform([])
            MultiChannelWaveform(None)

    def test_init_single_channel(self) -> None:
        dwf = DummyWaveform(duration=1.3)
        self.assertRaises(ValueError, MultiChannelWaveform, [(dwf, [1])])
        self.assertRaises(ValueError, MultiChannelWaveform, [(dwf, [-1])])
        waveform = MultiChannelWaveform([(dwf, [0])])
        self.assertEqual(1, waveform.num_channels)
        self.assertEqual(1.3, waveform.duration)

    def test_init_several_channels(self) -> None:
        dwfa = DummyWaveform(duration=4.2, num_channels=2)
        dwfb = DummyWaveform(duration=4.2, num_channels=3)
        dwfc = DummyWaveform(duration=2.3)
        self.assertRaises(ValueError, MultiChannelWaveform, [(dwfa, [2, 4]), (dwfb, [3, 5, 1])])
        self.assertRaises(ValueError, MultiChannelWaveform, [(dwfa, [0, 1]), (dwfc, [2])])
        waveform = MultiChannelWaveform([(dwfa, [2, 4]), (dwfb, [3, 0, 1])])
        self.assertEqual(5, waveform.num_channels)
        self.assertEqual(4.2, waveform.duration)

    def test_sample(self) -> None:
        sample_times = numpy.linspace(98.5, 103.5, num=11)
        samples_a = numpy.array([
            [0, 0.5], [1, 0.6], [2, 0.7], [3, 0.8], [4, 0.9], [5, 1.0],
            [6, 1.1], [7, 1.2], [8, 1.3], [9, 1.4], [10, 1.5]
        ])
        samples_b = numpy.array(
            [[-10], [-11], [-12], [-13], [-14], [-15], [-16], [-17], [-18], [-19], [-20]]
        )
        dwf_a = DummyWaveform(duration=3.2, sample_output=samples_a, num_channels=2)
        dwf_b = DummyWaveform(duration=3.2, sample_output=samples_b, num_channels=1)
        waveform = MultiChannelWaveform([(dwf_a, [2, 0]), (dwf_b, [1])])
        self.assertEqual(3, waveform.num_channels)
        self.assertEqual(3.2, waveform.duration)

        result = waveform.sample(sample_times, 0.7)
        self.assertEqual([(list(sample_times), 0.7)], dwf_a.sample_calls)
        self.assertEqual([(list(sample_times), 0.7)], dwf_b.sample_calls)

        expected = numpy.array([
            [0.5, -10, 0],
            [0.6, -11, 1],
            [0.7, -12, 2],
            [0.8, -13, 3],
            [0.9, -14, 4],
            [1.0, -15, 5],
            [1.1, -16, 6],
            [1.2, -17, 7],
            [1.3, -18, 8],
            [1.4, -19, 9],
            [1.5, -20, 10],
        ])
        self.assertTrue(numpy.all(expected == result))

    def test_equality(self) -> None:
        dwf_a = DummyWaveform(duration=246.2, num_channels=2)
        waveform_a1 = MultiChannelWaveform([(dwf_a, [0, 1])])
        waveform_a2 = MultiChannelWaveform([(dwf_a, [0, 1])])
        waveform_a3 = MultiChannelWaveform([(dwf_a, [1, 0])])
        self.assertEqual(waveform_a1, waveform_a1)
        self.assertEqual(waveform_a1, waveform_a2)
        self.assertNotEqual(waveform_a1, waveform_a3)


class MultiChannelPulseTemplateTest(unittest.TestCase):

    def test_build_sequence_no_params(self) -> None:
        dummy1 = DummyPulseTemplate(parameter_names={'foo'})
        pulse = MultiChannelPulseTemplate([(dummy1, {'foo': '2*bar'}, [1]),
                                           (dummy1, {'foo': '3'}, [0])], {'bar'})

        self.assertEqual({'bar'}, pulse.parameter_names)
        self.assertEqual({ParameterDeclaration('bar')}, pulse.parameter_declarations)

        self.assertRaises(ParameterNotProvidedException, pulse.build_waveform, {})

        self.assertRaises(ParameterNotProvidedException, pulse.build_sequence, DummySequencer(), {}, {}, DummyInstructionBlock())