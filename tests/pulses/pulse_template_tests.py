import unittest

from typing import Optional, Dict, Set, Any, List

from qctoolkit.pulses.pulse_template import AtomicPulseTemplate, MeasurementWindow
from qctoolkit.pulses.instructions import SingleChannelWaveform, EXECInstruction
from qctoolkit.pulses.parameters import Parameter, ParameterDeclaration
from qctoolkit.pulses.multi_channel_pulse_template import MultiChannelWaveform

from tests.pulses.sequencing_dummies import DummySingleChannelWaveform, DummySequencer, DummyInstructionBlock


class AtomicPulseTemplateStub(AtomicPulseTemplate):

    def is_interruptable(self) -> bool:
        return super().is_interruptable()

    def __init__(self, waveform: SingleChannelWaveform, measurement_windows: List[MeasurementWindow] = [],
                 identifier: Optional[str]=None) -> None:
        super().__init__(identifier=identifier)
        self.waveform = waveform
        self.measurement_windows = measurement_windows

    def build_waveform(self, parameters: Dict[str, Parameter]):
        return self.waveform

    def get_measurement_windows(self, parameters: Dict[str, Parameter] = None):
        return self.measurement_windows

    def requires_stop(self,
                      parameters: Dict[str, Parameter],
                      conditions: Dict[str, 'Condition']) -> bool:
        return False

    @property
    def defined_channels(self) -> Set['ChannelID']:
        raise NotImplementedError()

    @property
    def measurement_names(self):
        raise NotImplementedError()

    @property
    def parameter_declarations(self) -> Set[ParameterDeclaration]:
        raise NotImplementedError()

    @property
    def parameter_names(self) -> Set[str]:
        raise NotImplementedError()

    def get_serialization_data(self, serializer: 'Serializer') -> Dict[str, Any]:
        raise NotImplementedError()

    @staticmethod
    def deserialize(serializer: 'Serializer', **kwargs) -> 'AtomicPulseTemplateStub':
        raise NotImplementedError()


class AtomicPulseTemplateTests(unittest.TestCase):

    def test_is_interruptable(self) -> None:
        wf = DummySingleChannelWaveform()
        template = AtomicPulseTemplateStub(wf)
        self.assertFalse(template.is_interruptable())
        template = AtomicPulseTemplateStub(wf, identifier="arbg4")
        self.assertFalse(template.is_interruptable())

    def test_build_sequence_no_waveform(self) -> None:
        sequencer = DummySequencer()
        block = DummyInstructionBlock()

        template = AtomicPulseTemplateStub(None)
        template.build_sequence(sequencer, {}, {}, {}, {}, block)
        self.assertFalse(block.instructions)

    def test_build_sequence(self) -> None:
        single_wf = DummySingleChannelWaveform(duration=6)
        wf = MultiChannelWaveform(dict(A=single_wf))
        measurement_windows = [('M', 0, 5)]
        sequencer = DummySequencer()
        block = DummyInstructionBlock()

        template = AtomicPulseTemplateStub(wf,measurement_windows)
        template.build_sequence(sequencer, {}, {}, measurement_mapping={'M': 'M2'}, channel_mapping={'A': 'B'}, instruction_block=block)
        self.assertEqual(len(block.instructions), 1)
        self.assertIsInstance(block.instructions[0], EXECInstruction)
        self.assertEqual(block.instructions[0].waveform.defined_channels, {'B'})
        self.assertEqual(block.instructions[0].waveform['B'], single_wf)
        self.assertEqual(block.instructions[0].measurement_windows, [('M2', 0, 5)])