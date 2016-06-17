import unittest
import copy

from qctoolkit.expressions import Expression
from qctoolkit.pulses.table_pulse_template import TablePulseTemplate
from qctoolkit.pulses.sequence_pulse_template import SequencePulseTemplate, MissingMappingException, UnnecessaryMappingException, MissingParameterDeclarationException
from qctoolkit.pulses.parameters import ParameterDeclaration, ParameterNotProvidedException, ConstantParameter
from qctoolkit.pulses.multi_channel_pulse_template import MultiChannelPulseTemplate

from tests.pulses.sequencing_dummies import DummySequencer, DummyInstructionBlock, DummyPulseTemplate, DummyParameter, DummyNoValueParameter
from tests.serialization_dummies import DummySerializer


class MultiChannelWaveformTest(unittest.TestCase):

    pass


class MultiChannelPulseTemplateTest(unittest.TestCase):

    def test_build_sequence_no_params(self) -> None:
        dummy1 = DummyPulseTemplate(parameter_names={'foo'})
        pulse = MultiChannelPulseTemplate([(dummy1, {'foo': '2*bar'}, [1]),
                                           (dummy1, {'foo': '3'}, [0])], {'bar'})

        self.assertEqual({'bar'}, pulse.parameter_names)
        self.assertEqual({ParameterDeclaration('bar')}, pulse.parameter_declarations)

        self.assertRaises(ParameterNotProvidedException, pulse.build_waveform, {})

        self.assertRaises(ParameterNotProvidedException, pulse.build_sequence, DummySequencer(), {}, {}, DummyInstructionBlock())