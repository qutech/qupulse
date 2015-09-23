"""STANDARD LIBRARY IMPORTS"""
from typing import Tuple, List, Dict, Optional, Any
import os
import sys
import numpy

srcPath = os.path.dirname(os.path.abspath(__file__)).rsplit('tests',1)[0] + 'src'
sys.path.insert(0,srcPath)

"""LOCAL IMPORTS"""
from pulses.Instructions import Waveform
from pulses.Sequencer import Sequencer, InstructionBlock, SequencingHardwareInterface, SequencingElement
from pulses.Parameter import Parameter
from pulses.Interpolation import InterpolationStrategy


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

    def __init__(self, outerBlock: InstructionBlock = None) -> None:
        super().__init__(outerBlock)
        self.embedded_blocks = [] # type: Collection[InstructionBlock]

    def create_embedded_block(self) -> InstructionBlock:
        block = InstructionBlock(self)
        self.embedded_blocks.append(block)
        return block


class DummyWaveform(Waveform):

    @property
    def _compare_key(self) -> Any:
        return id(self)

    @property
    def duration(self) -> float:
        return 0

    def sample(self, sample_times: numpy.ndarray, first_offset: float=0) -> numpy.ndarray:
        raise NotImplementedError()


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


class DummyInterpolationStrategy(InterpolationStrategy):

    def __init__(self) -> None:
        self.call_arguments = []

    def __call__(self, start: Tuple[float, float], end: Tuple[float, float], times: numpy.ndarray) -> numpy.ndarray:
        self.call_arguments.append((start, end, list(times)))
        return times

    def __repr__(self) -> str:
        return "DummyInterpolationStrategy {}".format(id(self))