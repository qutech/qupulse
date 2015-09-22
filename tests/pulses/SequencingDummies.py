"""STANDARD LIBRARY IMPORTS"""
from typing import Tuple, List, Dict, Optional, Set, Any
import numpy

"""LOCAL IMPORTS"""
from pulses.Instructions import Waveform
from pulses.Sequencer import Sequencer, InstructionBlock, SequencingHardwareInterface, SequencingElement
from pulses.Parameter import Parameter, ParameterDeclaration
from pulses.Serializer import Serializer
from pulses.PulseTemplate import PulseTemplate, MeasurementWindow
from pulses.Interpolation import InterpolationStrategy
from pulses.Condition import Condition


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

    def get_serialization_data(self, serializer: Serializer) -> None:
            raise NotImplemented()

    @staticmethod
    def deserialize(serializer: Serializer) -> 'DummyParameter':
        raise NotImplemented()


class DummySequencingElement(SequencingElement):

    def __init__(self, requires_stop: bool = False, push_elements: Tuple[InstructionBlock, List[SequencingElement]] = None) -> None:
        super().__init__()
        self.build_call_counter = 0
        self.requires_stop_call_counter = 0
        self.target_block = None
        self.parameters = None
        self.requires_stop_ = requires_stop
        self.push_elements = push_elements
        self.parameter_names = set()

    def build_sequence(self, sequencer: Sequencer, parameters: Dict[str, Parameter], instruction_block: InstructionBlock) -> None:
        self.build_call_counter += 1
        self.target_block = instruction_block
        self.parameters = parameters
        if self.push_elements is not None:
            for element in self.push_elements[1]:
                sequencer.push(element, parameters, self.push_elements[0])

    def requires_stop(self, parameters: Dict[str, Parameter]) -> bool:
        self.requires_stop_call_counter += 1
        self.parameters = parameters
        return self.requires_stop_


class DummySequencingHardware(SequencingHardwareInterface):

    def __init__(self, sample_rate: float=1) -> None:
        super().__init__()
        self.waveforms = [] # type: List[WaveformTable]
        self.sample_rate_ = sample_rate

    def register_waveform(self, waveform: Waveform) -> None:
        self.waveforms.append(waveform)

    @property
    def sample_rate(self) -> float:
        return self.sample_rate_


class DummyInstructionBlock(InstructionBlock):

    def __init__(self, outer_block: InstructionBlock = None) -> None:
        super().__init__(outer_block)
        self.embedded_blocks = [] # type: Collection[InstructionBlock]

    def create_embedded_block(self) -> InstructionBlock:
        block = InstructionBlock(self)
        self.embedded_blocks.append(block)
        return block


class DummyWaveform(Waveform):

    def __init__(self, duration: float=0) -> None:
        super().__init__()
        self.duration_ = duration
        self.sample_calls = []

    @property
    def _compare_key(self) -> Any:
        return id(self)

    @property
    def duration(self) -> float:
        return self.duration_

    def sample(self, sample_times: numpy.ndarray, first_offset: float=0) -> numpy.ndarray:
        self.sample_calls.append((list(sample_times), first_offset))
        return sample_times


class DummySequencer(Sequencer):

    def __init__(self, sequencing_hardware: DummySequencingHardware) -> None:
        super().__init__(None)
        self.sequencing_stacks = {} #type: Dict[InstructionBlock, List[StackElement]]
        self.hardware = sequencing_hardware

    def push(self, sequencing_element: SequencingElement, parameters: Dict[str, Parameter], target_block: InstructionBlock = None) -> None:
        if target_block is None:
            target_block = self.__main_block

        if not target_block in self.sequencing_stacks:
            self.sequencing_stacks[target_block] = []

        self.sequencing_stacks[target_block].append((sequencing_element, parameters))

    def build(self) -> InstructionBlock:
        raise NotImplementedError()

    def has_finished(self):
        raise NotImplementedError()

    def register_waveform(self, waveform: Waveform) -> None:
        self.hardware.register_waveform(waveform)


class DummyPulseTemplate(PulseTemplate):

    def __init__(self, requires_stop: bool=False, is_interruptable: bool=False, parameter_names: Set[str]={}) -> None:
        super().__init__()
        self.requires_stop_ = requires_stop
        self.is_interruptable_ = is_interruptable
        self.parameter_names_ = parameter_names
        self.build_sequence_calls = 0

    @property
    def parameter_names(self) -> Set[str]:
        return self.parameter_names_

    @property
    def parameter_declarations(self) -> Set[str]:
        return [ParameterDeclaration(name) for name in self.parameter_names]

    def get_measurement_windows(self, parameters: Dict[str, Parameter] = None) -> List[MeasurementWindow]:
        """Return all measurement windows defined in this PulseTemplate."""
        raise NotImplementedError()

    @property
    def is_interruptable(self) -> bool:
        return self.is_interruptable_

    def build_sequence(self, sequencer: Sequencer, parameters: Dict[str, Parameter], instruction_block: InstructionBlock):
        self.build_sequence_calls += 1

    def requires_stop(self, parameters: Dict[str, Parameter]):
        return self.requires_stop_

    def get_serialization_data(self, serializer: Serializer):
        raise NotImplementedError()

    @staticmethod
    def deserialize(serializer: Serializer,
                    condition: Dict[str, Any],
                    body: Dict[str, Any],
                    identifier: Optional[str]=None):
        raise NotImplementedError()


class DummyInterpolationStrategy(InterpolationStrategy):

    def __init__(self) -> None:
        self.call_arguments = []

    def __call__(self, start: Tuple[float, float], end: Tuple[float, float], times: numpy.ndarray) -> numpy.ndarray:
        self.call_arguments.append((start, end, list(times)))
        return times

    def __repr__(self) -> str:
        return "DummyInterpolationStrategy {}".format(id(self))


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
                            instruction_block: InstructionBlock) -> None:
        self.loop_call_data = dict(
            delegator=delegator,
            body=body,
            sequencer=sequencer,
            parameters=parameters,
            instruction_block=instruction_block
        )

    def build_sequence_branch(self,
                              delegator: SequencingElement,
                              if_branch: SequencingElement,
                              else_branch: SequencingElement,
                              sequencer: Sequencer,
                              parameters: Dict[str, Parameter],
                              instruction_block: InstructionBlock) -> None:
        self.branch_call_data = dict(
            delegator=delegator,
            if_branch=if_branch,
            else_branch=else_branch,
            sequencer=sequencer,
            parameters=parameters,
            instruction_block=instruction_block
        )


class DummyPulseTemplate(PulseTemplate):

    def __init__(self, requires_stop: bool=False, is_interruptable: bool=False, parameter_names: Set[str]={}) -> None:
        super().__init__()
        self.requires_stop_ = requires_stop
        self.is_interruptable_ = is_interruptable
        self.parameter_names_ = parameter_names
        self.build_sequence_calls = 0

    @property
    def parameter_names(self) -> Set[str]:
        return self.parameter_names_

    @property
    def parameter_declarations(self) -> Set[str]:
        return [ParameterDeclaration(name) for name in self.parameter_names]

    def get_measurement_windows(self, parameters: Dict[str, Parameter] = None) -> List[MeasurementWindow]:
        """Return all measurement windows defined in this PulseTemplate."""
        raise NotImplementedError()

    @property
    def is_interruptable(self) -> bool:
        return self.is_interruptable_

    def build_sequence(self, sequencer: Sequencer, parameters: Dict[str, Parameter], instruction_block: InstructionBlock):
        self.build_sequence_calls += 1

    def requires_stop(self, parameters: Dict[str, Parameter]):
        return self.requires_stop_

    def get_serialization_data(self, serializer: Serializer):
        raise NotImplementedError()

    @staticmethod
    def deserialize(serializer: Serializer,
                    condition: Dict[str, Any],
                    body: Dict[str, Any],
                    identifier: Optional[str]=None):
        raise NotImplementedError()