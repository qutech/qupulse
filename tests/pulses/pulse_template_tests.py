import unittest

from typing import Optional, Dict, Set, Any, List

from qctoolkit.pulses.pulse_template import AtomicPulseTemplate, MeasurementWindow
from qctoolkit.pulses.instructions import Waveform, EXECInstruction
from qctoolkit.pulses.parameters import Parameter, ParameterDeclaration

from tests.pulses.sequencing_dummies import DummyWaveform, DummySequencer, DummyInstructionBlock


class AtomicPulseTemplateStub(AtomicPulseTemplate):

    def is_interruptable(self) -> bool:
        return super().is_interruptable()

    def __init__(self, waveform: Waveform, measurement_windows: List[MeasurementWindow] = [],
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
    def num_channels(self) -> int:
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
        wf = DummyWaveform()
        template = AtomicPulseTemplateStub(wf)
        self.assertFalse(template.is_interruptable())
        template = AtomicPulseTemplateStub(wf, identifier="arbg4")
        self.assertFalse(template.is_interruptable())

    def test_build_sequence_no_waveform(self) -> None:
        sequencer = DummySequencer()
        block = DummyInstructionBlock()

        template = AtomicPulseTemplateStub(None)
        template.build_sequence(sequencer, {}, {}, {}, block)
        self.assertFalse(block.instructions)

    def test_build_sequence(self) -> None:
        wf = DummyWaveform()
        sequencer = DummySequencer()
        block = DummyInstructionBlock()

        template = AtomicPulseTemplateStub(wf)
        template.build_sequence(sequencer, {}, {}, {}, block)
        self.assertEqual([EXECInstruction(wf)], block.instructions)