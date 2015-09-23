"""STANDARD LIBRARY IMPORTS"""
from typing import Tuple, List, Dict, Optional, Set, Any

"""LOCAL IMPORTS"""
from pulses.Instructions import WaveformTable, Waveform
from pulses.Sequencer import Sequencer, InstructionBlock, SequencingHardwareInterface, SequencingElement
from pulses.Parameter import Parameter, ParameterDeclaration
from pulses.Serializer import Serializer
from pulses.PulseTemplate import PulseTemplate, MeasurementWindow


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

    def __init__(self) -> None:
        super().__init__()
        self.waveforms = [] # type: List[WaveformTable]


    def register_waveform(self, waveform_table: WaveformTable) -> Waveform:
        self.waveforms.append(waveform_table)
        return DummyWaveform(waveform_table)

class DummyInstructionBlock(InstructionBlock):

    def __init__(self, outerBlock: InstructionBlock = None) -> None:
        super().__init__(outerBlock)
        self.embedded_blocks = [] # type: Collection[InstructionBlock]

    def create_embedded_block(self) -> InstructionBlock:
        block = InstructionBlock(self)
        self.embedded_blocks.append(block)
        return block

class DummyWaveform(Waveform):

    def __init__(self, waveform_table: WaveformTable) -> None:
        super().__init__(len(waveform_table))
        self.waveform_table = waveform_table

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

    def register_waveform(self, waveform_table: WaveformTable) -> Waveform:
        return self.hardware.register_waveform(waveform_table)


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