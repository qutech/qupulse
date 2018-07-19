import unittest

from typing import Optional, Dict, Set, Any

from qctoolkit.utils.types import ChannelID
from qctoolkit.expressions import Expression, ExpressionScalar
from qctoolkit.pulses.pulse_template import AtomicPulseTemplate, PulseTemplate
from qctoolkit._program.instructions import Waveform, EXECInstruction, MEASInstruction
from qctoolkit.pulses.parameters import Parameter, ConstantParameter
from qctoolkit.pulses.multi_channel_pulse_template import MultiChannelWaveform

from tests.pulses.sequencing_dummies import DummyWaveform, DummySequencer, DummyInstructionBlock


class PulseTemplateStub(PulseTemplate):
    def __init__(self, identifier=None,
                 defined_channels=None,
                 duration=None,
                 parameter_names=None,
                 measurement_names=None,
                 registry=None):
        super().__init__(identifier=identifier)

        self._defined_channels = defined_channels
        self._duration = duration
        self._parameter_names = parameter_names
        self._measurement_names = set() if measurement_names is None else measurement_names
        self.internal_create_program_args = []
        self._register(registry=registry)

    @property
    def defined_channels(self) -> Set[ChannelID]:
        if self._defined_channels:
            return self._defined_channels
        else:
            raise NotImplementedError()

    @property
    def parameter_names(self) -> Set[str]:
        if self._parameter_names is None:
            raise NotImplementedError()
        return self._parameter_names

    def get_serialization_data(self, serializer: Optional['Serializer']=None) -> Dict[str, Any]:
        raise NotImplementedError()

    @staticmethod
    def deserialize(serializer: Optional['Serializer']=None, **kwargs) -> 'AtomicPulseTemplateStub':
        raise NotImplementedError()

    @property
    def duration(self) -> Expression:
        if self._duration is None:
            raise NotImplementedError()
        return self._duration

    def build_sequence(self,
                       sequencer: "Sequencer",
                       parameters: Dict[str, Parameter],
                       conditions: Dict[str, 'Condition'],
                       measurement_mapping: Dict[str, str],
                       channel_mapping: Dict['ChannelID', 'ChannelID'],
                       instruction_block: 'InstructionBlock'):
        raise NotImplementedError()

    def _internal_create_program(self,
                                 parameters: Dict[str, Parameter],
                                 measurement_mapping: Dict[str, Optional[str]],
                                 channel_mapping: Dict[ChannelID, Optional[ChannelID]]) -> Optional['Loop']:
        self.internal_create_program_args.append((parameters, measurement_mapping, channel_mapping))
        return None

    def is_interruptable(self):
        raise NotImplementedError()

    @property
    def measurement_names(self):
        return self._measurement_names

    def requires_stop(self,
                      parameters: Dict[str, Parameter],
                      conditions: Dict[str, 'Condition']):
        raise NotImplementedError()

    @property
    def integral(self) -> Dict[ChannelID, ExpressionScalar]:
        raise NotImplementedError()


class AtomicPulseTemplateStub(AtomicPulseTemplate):
    def is_interruptable(self) -> bool:
        return super().is_interruptable()

    def __init__(self, *, waveform: Waveform=None, duration: Expression=None, measurements=None,
                 parameter_names: Optional[Set] = None, identifier: Optional[str]=None,
                 registry=None) -> None:
        super().__init__(identifier=identifier, measurements=measurements)
        self.waveform = waveform
        self._duration = duration
        self._parameter_names = parameter_names
        self.retrieved_parameters = []
        self.retrieved_channel_mapping = []
        self._register(registry=registry)

    def build_waveform(self, parameters: Dict[str, Parameter], channel_mapping):
        self.retrieved_parameters.append(parameters)
        self.retrieved_channel_mapping.append(channel_mapping)
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
        if self._parameter_names is None:
            raise NotImplementedError()
        return self._parameter_names

    def get_serialization_data(self, serializer: Optional['Serializer']=None) -> Dict[str, Any]:
        raise NotImplementedError()

    @property
    def measurement_names(self):
        raise NotImplementedError()

    @classmethod
    def deserialize(cls, serializer: Optional['Serializer']=None, **kwargs) -> 'AtomicPulseTemplateStub':
        raise NotImplementedError()

    @property
    def duration(self) -> Expression:
        return self._duration

    @property
    def integral(self) -> Dict[ChannelID, ExpressionScalar]:
        raise NotImplementedError()


class PulseTemplateTest(unittest.TestCase):

    def test_create_program(self) -> None:
        template = PulseTemplateStub(defined_channels={'A'}, parameter_names={'foo'})
        parameters = {'foo': ConstantParameter(2.126), 'bar': -26.2, 'hugo': '2*x+b'}
        measurement_mapping = {'M': 'N'}
        channel_mapping = {'B': 'A'}
        template.create_program(parameters=parameters,
                                measurement_mapping=measurement_mapping,
                                channel_mapping=channel_mapping)
        expected_parameters = {'foo': ConstantParameter(2.126), 'bar': ConstantParameter(-26.2), 'hugo': ConstantParameter('2*x+b')}
        self.assertEqual((expected_parameters, measurement_mapping, channel_mapping), template.internal_create_program_args[-1])


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
        single_wf = DummyWaveform(duration=6, defined_channels={'A'})
        wf = MultiChannelWaveform([single_wf])

        sequencer = DummySequencer()
        block = DummyInstructionBlock()

        template = AtomicPulseTemplateStub(waveform=wf, measurements=measurement_windows)
        template.build_sequence(sequencer, {}, {}, measurement_mapping={'M': 'N'}, channel_mapping={},
                                instruction_block=block)
        self.assertEqual(len(block.instructions), 2)

        meas, exec = block.instructions
        self.assertIsInstance(meas, MEASInstruction)
        self.assertEqual(meas.measurements, [('N', 0, 5)])

        self.assertIsInstance(exec, EXECInstruction)
        self.assertEqual(exec.waveform.defined_channels, {'A'})

    def test_internal_create_program(self) -> None:
        measurement_windows = [('M', 0, 5)]
        single_wf = DummyWaveform(duration=6, defined_channels={'A'})
        wf = MultiChannelWaveform([single_wf])

        template = AtomicPulseTemplateStub(waveform=wf, measurements=measurement_windows, parameter_names={'foo'})
        parameters = {'foo': ConstantParameter(7.2)}
        channel_mapping = {'B': 'A'}
        program = template._internal_create_program(parameters=parameters,
                                                    measurement_mapping={'M': 'N'},
                                                    channel_mapping=channel_mapping)
        self.assertEqual({k: p.get_value() for k, p in parameters.items()}, template.retrieved_parameters[-1])
        self.assertIs(channel_mapping, template.retrieved_channel_mapping[-1])
        expected_measurement_windows = {'N': (0, 5)}
        self.assertIs(program.waveform, wf)
        self.assertEqual(expected_measurement_windows, program.get_measurement_windows())
