import unittest

import copy
from typing import Optional, Dict, Set, Any, List

from qctoolkit.utils.types import MeasurementWindow, ChannelID
from qctoolkit.expressions import Expression
from qctoolkit.pulses.pulse_template import AtomicPulseTemplate, MeasurementDeclaration, PulseTemplate
from qctoolkit.pulses.instructions import Waveform, EXECInstruction
from qctoolkit.pulses.parameters import Parameter
from qctoolkit.pulses.multi_channel_pulse_template import MultiChannelWaveform

from tests.pulses.sequencing_dummies import DummyWaveform, DummySequencer, DummyInstructionBlock


class PulseTemplateStub(PulseTemplate):
    def __init__(self, identifier=None,
                 defined_channels=None,
                 duration=None,
                 parameter_names=None,
                 measurement_names=None):
        super().__init__(identifier=identifier)

        self._defined_channels = defined_channels
        self._duration = duration
        self._parameter_names = parameter_names
        self._measurement_names = set() if measurement_names is None else measurement_names

    @property
    def defined_channels(self) -> Set['ChannelID']:
        if self._defined_channels:
            return self._defined_channels
        else:
            raise NotImplementedError()

    @property
    def parameter_names(self) -> Set[str]:
        if self._parameter_names is None:
            raise NotImplementedError()
        return self._parameter_names

    def get_serialization_data(self, serializer: 'Serializer') -> Dict[str, Any]:
        raise NotImplementedError()

    @staticmethod
    def deserialize(serializer: 'Serializer', **kwargs) -> 'AtomicPulseTemplateStub':
        raise NotImplementedError()

    @property
    def duration(self) -> Expression:
        if self._duration is None:
            raise NotImplementedError()
        return self._duration

    def get_serialization_data(self, serializer: 'Serializer') -> Dict[str, Any]:
        raise NotImplementedError()

    @staticmethod
    def deserialize(serializer: 'Serializer', **kwargs) -> 'AtomicPulseTemplateStub':
        raise NotImplementedError()

    def build_sequence(self,
                       sequencer: "Sequencer",
                       parameters: Dict[str, Parameter],
                       conditions: Dict[str, 'Condition'],
                       measurement_mapping: Dict[str, str],
                       channel_mapping: Dict['ChannelID', 'ChannelID'],
                       instruction_block: 'InstructionBlock'):
        raise NotImplementedError()

    def is_interruptable(self):
        raise NotImplementedError()

    @property
    def measurement_names(self):
        return self._measurement_names

    def requires_stop(self,
                      parameters: Dict[str, Parameter],
                      conditions: Dict[str, 'Condition']):
        raise NotImplementedError()


class AtomicPulseTemplateStub(AtomicPulseTemplate):
    def is_interruptable(self) -> bool:
        return super().is_interruptable()

    def __init__(self, *, waveform: Waveform=None, duration: Expression=None,
                 identifier: Optional[str]=None) -> None:
        super().__init__(identifier=identifier)
        self.waveform = waveform
        self._duration = duration

    def build_waveform(self, parameters: Dict[str, Parameter], measurement_mapping, channel_mapping):
        return self.waveform

    def requires_stop(self,
                      parameters: Dict[str, Parameter],
                      conditions: Dict[str, 'Condition']) -> bool:
        return False

    @property
    def defined_channels(self) -> Set['ChannelID']:
        raise NotImplementedError()

    @property
    def parameter_names(self) -> Set[str]:
        raise NotImplementedError()

    def get_serialization_data(self, serializer: 'Serializer') -> Dict[str, Any]:
        raise NotImplementedError()

    @property
    def measurement_names(self):
        raise NotImplementedError()

    @staticmethod
    def deserialize(serializer: 'Serializer', **kwargs) -> 'AtomicPulseTemplateStub':
        raise NotImplementedError()

    @property
    def duration(self) -> Expression:
        return self._duration


class AtomicPulseTemplateTests(unittest.TestCase):

    def test_is_interruptable(self) -> None:
        wf = DummyWaveform()
        template = AtomicPulseTemplateStub(waveform=wf)
        self.assertFalse(template.is_interruptable())
        template = AtomicPulseTemplateStub(waveform=wf, identifier="arbg4")
        self.assertFalse(template.is_interruptable())

    def test_build_sequence_no_waveform(self) -> None:
        sequencer = DummySequencer()
        block = DummyInstructionBlock()

        template = AtomicPulseTemplateStub()
        template.build_sequence(sequencer, {}, {}, {}, {}, block)
        self.assertFalse(block.instructions)

    def test_build_sequence(self) -> None:
        measurement_windows = [('M', 0, 5)]
        single_wf = DummyWaveform(duration=6, defined_channels={'A'}, measurement_windows=measurement_windows)
        wf = MultiChannelWaveform([single_wf])

        sequencer = DummySequencer()
        block = DummyInstructionBlock()

        template = AtomicPulseTemplateStub(waveform=wf)
        template.build_sequence(sequencer, {}, {}, measurement_mapping={}, channel_mapping={}, instruction_block=block)
        self.assertEqual(len(block.instructions), 1)
        self.assertIsInstance(block.instructions[0], EXECInstruction)
        self.assertEqual(block.instructions[0].waveform.defined_channels, {'A'})
        self.assertEqual(list(block.instructions[0].waveform.get_measurement_windows()), [('M', 0, 5)])

