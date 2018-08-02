"""STANDARD LIBRARY IMPORTS"""
from typing import Tuple, List, Dict, Optional, Set, Any, Union
import copy

import numpy

"""LOCAL IMPORTS"""
from qctoolkit.utils.types import MeasurementWindow, ChannelID, TimeType, time_from_float
from qctoolkit.serialization import Serializer
from qctoolkit._program.waveforms import Waveform
from qctoolkit._program.instructions import Instruction, CJMPInstruction, GOTOInstruction, REPJInstruction
from qctoolkit.pulses.sequencing import Sequencer, InstructionBlock, SequencingElement
from qctoolkit.pulses.parameters import Parameter
from qctoolkit.pulses.pulse_template import AtomicPulseTemplate
from qctoolkit.pulses.interpolation import InterpolationStrategy
from qctoolkit.pulses.conditions import Condition
from qctoolkit.expressions import Expression, ExpressionScalar

class DummyParameter(Parameter):

    def __init__(self, value: float = 0, requires_stop: bool = False) -> None:
        super().__init__()
        self.value = value
        self.requires_stop_ = requires_stop

    def get_value(self) -> float:
        return self.value

    @property
    def requires_stop(self) -> bool:
        return self.requires_stop_

    def __hash__(self):
        return hash(self.value)

    def get_serialization_data(self, serializer: Optional[Serializer]=None) -> None:
            raise NotImplementedError()

    @classmethod
    def deserialize(cls, serializer: Optional[Serializer]=None) -> 'DummyParameter':
        raise NotImplementedError()

class DummyNoValueParameter(Parameter):

    def __init__(self) -> None:
        super().__init__()

    def get_value(self) -> float:
        raise Exception("May not call get_value on DummyNoValueParameter.")

    @property
    def requires_stop(self) -> bool:
        return True

    def get_serialization_data(self, serializer: Optional[Serializer]=None) -> None:
            raise NotImplementedError()

    @classmethod
    def deserialize(cls, serializer: Optional[Serializer]=None) -> 'DummyParameter':
        raise NotImplementedError()

    def __hash__(self):
        return 0

class DummySequencingElement(SequencingElement):

    def __init__(self, requires_stop: bool = False, push_elements: Tuple[InstructionBlock, List[SequencingElement]] = None) -> None:
        super().__init__()
        self.build_call_counter = 0
        self.requires_stop_call_counter = 0
        self.target_block = None
        self.parameters = None
        self.conditions = None
        self.window_mapping = None
        self.channel_mapping =  None
        self.requires_stop_ = requires_stop
        self.push_elements = push_elements
        self.parameter_names = set()
        self.condition_names = set()
        self.atomicity_ = False

    def build_sequence(self,
                       sequencer: Sequencer,
                       parameters: Dict[str, Parameter],
                       conditions: Dict[str, 'Condition'],
                       measurement_mapping: Optional[Dict[str, str]],
                       channel_mapping: Dict['ChannelID', 'ChannelID'],
                       instruction_block: InstructionBlock) -> None:
        self.build_call_counter += 1
        self.target_block = instruction_block
        instruction_block.add_instruction(DummyInstruction(self))
        self.parameters = parameters
        self.conditions = conditions
        self.window_mapping = measurement_mapping
        self.channel_mapping = channel_mapping
        if self.push_elements is not None:
            for element in self.push_elements[1]:
                sequencer.push(element, parameters, conditions,
                               window_mapping=measurement_mapping,
                               channel_mapping=channel_mapping,
                               target_block=self.push_elements[0])

    def requires_stop(self, parameters: Dict[str, Parameter], conditions: Dict[str, 'Conditions']) -> bool:
        self.requires_stop_call_counter += 1
        self.parameters = parameters
        self.conditions = conditions
        return self.requires_stop_

    @property
    def atomicity(self):
        return self.atomicity_


class DummyInstruction(Instruction):

    def __init__(self, elem: DummySequencingElement = None) -> None:
        super().__init__()
        self.elem = elem

    @property
    def compare_key(self) -> Any:
        return self.elem


class DummyInstructionBlock(InstructionBlock):

    def __init__(self) -> None:
        super().__init__()
        self.embedded_blocks = [] # type: Collection[InstructionBlock]

    def add_instruction(self, instruction: Instruction) -> None:
        super().add_instruction(instruction)
        if isinstance(instruction, (CJMPInstruction, GOTOInstruction, REPJInstruction)):
            self.embedded_blocks.append(instruction.target.block)


class DummyWaveform(Waveform):

    def __init__(self, duration: float=0, sample_output: Union[numpy.ndarray, dict]=None, defined_channels={'A'}) -> None:
        super().__init__()
        self.duration_ = time_from_float(duration)
        self.sample_output = sample_output
        self.defined_channels_ = defined_channels
        self.sample_calls = []

    @property
    def compare_key(self) -> Any:
        if self.sample_output is not None:
            return hash(bytes(self.sample_output))
        else:
            return id(self)

    @property
    def duration(self) -> TimeType:
        return self.duration_

    @property
    def measurement_windows(self):
        return []

    def unsafe_sample(self,
                      channel: ChannelID,
                      sample_times: numpy.ndarray,
                      output_array: numpy.ndarray = None) -> numpy.ndarray:
        self.sample_calls.append((channel, list(sample_times), output_array))
        if output_array is None:
            output_array = numpy.empty_like(sample_times)
        if self.sample_output is not None:
            if isinstance(self.sample_output, dict):
                output_array[:] = self.sample_output[channel]
            else:
                output_array[:] = self.sample_output
        else:
            output_array[:] = sample_times
        return output_array

    def unsafe_get_subset_for_channels(self, channels: Set[ChannelID]) -> 'Waveform':
        if not channels <= self.defined_channels_:
            raise KeyError('channels not in defined_channels')
        c = copy.copy(self)
        c.defined_channels_ = channels
        return c

    @property
    def defined_channels(self):
        return self.defined_channels_


class DummySequencer(Sequencer):

    def __init__(self) -> None:
        super().__init__()
        self.sequencing_stacks = {} #type: Dict[InstructionBlock, List[StackElement]]

    def push(self,
             sequencing_element: SequencingElement,
             parameters: Dict[str, Parameter],
             conditions: Dict[str, 'Condition'],
             window_mapping: Optional[Dict[str, str]] = None,
             channel_mapping: Dict['ChannelID', 'ChannelID'] = dict(),
             target_block: InstructionBlock = None) -> None:
        if target_block is None:
            target_block = self.__main_block

        if target_block not in self.sequencing_stacks:
            self.sequencing_stacks[target_block] = []

        self.sequencing_stacks[target_block].append((sequencing_element, parameters, conditions, window_mapping,
                                                     channel_mapping))

    def build(self) -> InstructionBlock:
        raise NotImplementedError()

    def has_finished(self):
        raise NotImplementedError()


class DummyInterpolationStrategy(InterpolationStrategy):

    def __init__(self) -> None:
        self.call_arguments = []

    def __call__(self, start: Tuple[float, float], end: Tuple[float, float], times: numpy.ndarray) -> numpy.ndarray:
        self.call_arguments.append((start, end, list(times)))
        return times

    def __repr__(self) -> str:
        return "DummyInterpolationStrategy {}".format(id(self))

    @property
    def integral(self) -> ExpressionScalar:
        raise NotImplementedError()

    @property
    def expression(self) -> ExpressionScalar:
        raise NotImplementedError()


class DummyCondition(Condition):

    def __init__(self, requires_stop: bool=False):
        super().__init__()
        self.requires_stop_ = requires_stop
        self.loop_call_data = {}
        self.branch_call_data = {}

    def requires_stop(self) -> bool:
        return self.requires_stop_

    def build_sequence_loop(self,
                            delegator: SequencingElement,
                            body: SequencingElement,
                            sequencer: Sequencer,
                            parameters: Dict[str, Parameter],
                            conditions: Dict[str, Condition],
                            measurement_mapping: Dict[str, str],
                            channel_mapping: Dict['ChannelID', 'ChannelID'],
                            instruction_block: InstructionBlock) -> None:
        self.loop_call_data = dict(
            delegator=delegator,
            body=body,
            sequencer=sequencer,
            parameters=parameters,
            conditions=conditions,
            measurement_mapping=measurement_mapping,
            channel_mapping=channel_mapping,
            instruction_block=instruction_block
        )

    def build_sequence_branch(self,
                              delegator: SequencingElement,
                              if_branch: SequencingElement,
                              else_branch: SequencingElement,
                              sequencer: Sequencer,
                              parameters: Dict[str, Parameter],
                              conditions: Dict[str, Condition],
                              measurement_mapping: Dict[str, str],
                              channel_mapping: Dict['ChannelID', 'ChannelID'],
                              instruction_block: InstructionBlock) -> None:
        self.branch_call_data = dict(
            delegator=delegator,
            if_branch=if_branch,
            else_branch=else_branch,
            sequencer=sequencer,
            parameters=parameters,
            conditions=conditions,
            measurement_mapping=measurement_mapping,
            channel_mapping=channel_mapping,
            instruction_block=instruction_block
        )


class DummyPulseTemplate(AtomicPulseTemplate):

    def __init__(self,
                 requires_stop: bool=False,
                 is_interruptable: bool=False,
                 parameter_names: Set[str]={},
                 defined_channels: Set[ChannelID]={'default'},
                 duration: Any=0,
                 waveform: Waveform=tuple(),
                 measurement_names: Set[str] = set(),
                 measurements: list=list(),
                 integrals: Dict[ChannelID, ExpressionScalar]={'default': ExpressionScalar(0)},
                 identifier=None,
                 registry=None) -> None:
        super().__init__(identifier=identifier, measurements=measurements)
        self.requires_stop_ = requires_stop
        self.requires_stop_arguments = []

        self.is_interruptable_ = is_interruptable
        self.parameter_names_ = parameter_names
        self.build_sequence_arguments = []
        self.defined_channels_ = defined_channels
        self._duration = Expression(duration)
        self.waveform = waveform
        self.build_waveform_calls = []
        self.measurement_names_ = set(measurement_names)
        self._integrals = integrals
        self._register(registry=registry)

    @property
    def duration(self):
        return self._duration

    @property
    def parameter_names(self) -> Set[str]:
        return set(self.parameter_names_)

    @property
    def build_sequence_calls(self):
        return len(self.build_sequence_arguments)

    @property
    def is_interruptable(self) -> bool:
        return self.is_interruptable_

    @property
    def defined_channels(self) -> Set[ChannelID]:
        return set(self.defined_channels_)

    @property
    def measurement_names(self) -> Set[str]:
        return self.measurement_names_

    def build_sequence(self,
                       sequencer: Sequencer,
                       parameters: Dict[str, Parameter],
                       conditions: Dict[str, Condition],
                       measurement_mapping: Dict[str, str],
                       channel_mapping: Dict['ChannelID', 'ChannelID'],
                       instruction_block: InstructionBlock):
        self.build_sequence_arguments.append((sequencer,parameters,conditions, measurement_mapping, channel_mapping, instruction_block))

    def build_waveform(self,
                       parameters: Dict[str, Parameter],
                       channel_mapping: Dict[ChannelID, ChannelID]):
        self.build_waveform_calls.append((parameters, channel_mapping))
        if self.waveform or self.waveform is None:
            return self.waveform
        return DummyWaveform(duration=self.duration, defined_channels=self.defined_channels)

    def requires_stop(self, parameters: Dict[str, Parameter], conditions: Dict[str, Condition]) -> bool:
        self.requires_stop_arguments.append((parameters,conditions))
        return self.requires_stop_

    def get_serialization_data(self, serializer: Optional['Serializer']=None) -> Dict[str, Any]:
        data = super().get_serialization_data(serializer=serializer)
        if serializer: # compatibility with old serialization routines
            data = dict()
        data['requires_stop'] = self.requires_stop_
        data['is_interruptable'] = self.is_interruptable
        data['parameter_names'] = self.parameter_names
        data['defined_channels'] = self.defined_channels
        data['duration'] = self.duration
        data['measurement_names'] = self.measurement_names
        data['integrals'] = self.integral
        return data

    @property
    def integral(self) -> Dict[ChannelID, ExpressionScalar]:
        return self._integrals

    @property
    def compare_key(self) -> Tuple[Any]:
        return (self.requires_stop_, self.is_interruptable, self.parameter_names,
                self.defined_channels, self.duration, self.waveform, self.measurement_names, self.integral)

    def __eq__(self, other) -> bool:
        if not isinstance(other, DummyPulseTemplate):
            return False
        return self.compare_key == other.compare_key
