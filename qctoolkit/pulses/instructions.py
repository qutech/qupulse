"""This module defines the abstract hardware instruction model of the qc-toolkit.

Classes:
    - Waveform: An instantiated pulse which can be sampled to a raw voltage value array.
    - Trigger: Representation of a hardware trigger.
    - WaveformSequence: A sequence of waveforms.
    - MutableWaveformSequence: A mutable WaveformSequence where new waveforms can be appended.
    - ImmutableWaveformSequence: An immutable WaveformSequence.
    - Instruction: Base class for hardware instructions.
    - CJMPInstruction: Conditional jump instruction.
    - EXECInstruction: Instruction to execute a waveform.
    - GOTOInstruction: Unconditional jump instruction.
    - STOPInstruction: Instruction which indicates the end of execution.
    - InstructionBlock: A block of instructions which are not yet embedded in a global sequence.
    - InstructionSequence: A single final sequence of instructions.
    - InstructionPointer: References an instruction's location in a sequence.
    - InstructionBlockNotYetPlacedException
    - InstructionBlockAlreadyFinalizedException
    - MissingReturnAddressException
"""

from abc import ABCMeta, abstractmethod, abstractproperty
from typing import List, Any, Optional, Iterable
import numpy

from qctoolkit.comparable import Comparable

# TODO lumip: add docstrings to InstructionBlock after issue #116 is resolved

__all__ = ["Waveform", "Trigger",
           "WaveformSequence", "MutableWaveformSequence", "ImmutableWaveformSequence",
           "InstructionPointer", "Instruction", "CJMPInstruction", "EXECInstruction",
           "GOTOInstruction", "STOPInstruction", "InstructionBlock", "InstructionSequence",
           "InstructionBlockNotYetPlacedException", "InstructionBlockAlreadyFinalizedException",
           "MissingReturnAddressException"
          ]


class Waveform(Comparable, metaclass=ABCMeta):
    """Represents an instantiated PulseTemplate which can be sampled to retrieve arrays of voltage
    values for the hardware."""

    @abstractproperty
    def duration(self) -> float:
        """The duration of the waveform in time units."""

    @abstractmethod
    def sample(self, sample_times: numpy.ndarray, first_offset: float=0) -> numpy.ndarray:
        """Sample the waveform at given sample times.

        The only requirement on the provided sample times is that they must be monotonously
        increasing. The must not lie in the range of [0, waveform.duration] (but will be normalized
        internally into that range for the sampling). For example, if this Waveform had a duration
        of 5 and the given sample times would be [11, 15, 20], the result would be the samples of
        this Waveform at [0, 2.5, 5] in the Waveforms domain. This allows easier sampling of
        multiple subsequent Waveforms.

        Args:
            numpy.ndarray sample_times: Times at which this Waveform will be sampled. Will be
                normalized such that they lie in the range [0, waveform.duration] for interpolation.
            float first_offset: Offset of the discrete first sample from the actual beginning of
                the waveform in a continuous time domain.
        Result:
            numpy.ndarray of the sampled values of this Waveform at the provided sample times.
        """


class Trigger(Comparable):
    """Abstract representation of a hardware trigger for hardware based branching decisions."""
        
    def __init__(self) -> None:
        super().__init__()

    @property
    def compare_key(self) -> Any:
        return id(self)
    
    def __str__(self) -> str:
        return "Trigger {}".format(hash(self))


class InstructionPointer(Comparable):
    """Reference to the location of an InstructionBlock.
    """
    
    def __init__(self, block: 'InstructionBlock', offset: int=0) -> None:
        super().__init__()
        if offset < 0:
            raise ValueError("offset must be a non-negative integer (was {})".format(offset))
        self.__block = block
        self.__offset = offset

    @property
    def block(self) -> 'InstructionBlock':
        return self.__block  # todo: immutable ?

    @property
    def offset(self) -> int:
        return self.__offset

    @property
    def compare_key(self) -> Any:
        return id(self.__block), self.__offset
        
    def __str__(self) -> str:
        return "IP:{0}#{1}".format(self.__block, self.__offset)
        # try:
        #     return "{}".format(self.get_absolute_address())
        # except InstructionBlockNotYetPlacedException:
        #     return "IP:{0}#{1}".format(self.__block, self.__offset)


class Instruction(Comparable, metaclass=ABCMeta):
    """A hardware instruction."""

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def compare_key(self) -> Any:
        pass


class CJMPInstruction(Instruction):
    """A conditional jump hardware instruction.

    Will cause the execution to jump to the instruction indicated by the InstructionPointer held
    by this CJMPInstruction if the given Trigger was fired. If not, this Instruction will have no
    effect, the execution will continue with the following.
    """

    def __init__(self, trigger: Trigger, target: InstructionPointer) -> None:
        super().__init__()
        self.trigger = trigger
        self.target = target

    @property
    def compare_key(self) -> Any:
        return self.trigger, self.target
        
    def __str__(self) -> str:
        return "cjmp to {} on {}".format(self.target, self.trigger)


class GOTOInstruction(Instruction):
    """An unconditional jump hardware instruction.

    Will cause the execution to jump to the instruction indicated by the InstructionPointer
    held by this GOTOInstruction.
    """
    
    def __init__(self, target: InstructionPointer) -> None:
        super().__init__()
        self.target = target

    @property
    def compare_key(self) -> Any:
        return self.target

    def __str__(self) -> str:
        return "goto to {}".format(self.target)


class EXECInstruction(Instruction):
    """An instruction to execute/play back a waveform."""

    def __init__(self, waveform: Waveform) -> None:
        super().__init__()
        self.waveform = waveform

    @property
    def compare_key(self) -> Any:
        return self.waveform

    def __str__(self) -> str:
        return "exec {}".format(self.waveform)


class STOPInstruction(Instruction):
    """An instruction which indicates the end of the program."""

    def __init__(self) -> None:
        super().__init__()

    @property
    def compare_key(self) -> Any:
        return 0

    def __str__(self) -> str:
        return "stop"
        
        
class InstructionBlockAlreadyFinalizedException(Exception):
    """Indicates that an attempt was made to change an already finalized InstructionBlock."""
    def __str__(self) -> str:
        return "An attempt was made to change an already finalized InstructionBlock."
        
        
class InstructionBlockNotYetPlacedException(Exception):
    """Indicates that an attempt was made to obtain the start address of an InstructionBlock that
    was not yet placed inside the corresponding outer block."""
    def __str__(self) -> str:
        return "An attempt was made to obtain the start address of an InstructionBlock that was " \
               "not yet finally placed inside the corresponding outer block."


class MissingReturnAddressException(Exception):
    """Indicates that an inner InstructionBlock has no return address."""
    def __str__(self) -> str:
        return "No return address is set!"
        
        
InstructionSequence = List[Instruction] # pylint: disable=invalid-name,invalid-sequence-index


# class InstructionBlock(Comparable, metaclass=ABCMeta):
#
#     def __init__(self) -> None:
#         super().__init__()
#
#     @abstractproperty
#     def instructions(self) -> List[Instruction]:
#         pass
#
#     @abstractproperty
#     def return_ip(self) -> InstructionPointer:
#         pass
#
#     @property
#     def compare_key(self) -> Any:
#         return id(self)
#
#     def str(self) -> str:
#         return str(hash(self))


class InstructionBlock(Comparable):
    
    def __init__(self) -> None:
        super().__init__()
        self.__instruction_list = [] # type: InstructionSequence
        self.__embedded_blocks = [] # type: List[InstructionBlock]

        self.__return_ip = None
        
    def add_instruction(self, instruction: Instruction) -> None:
        self.__instruction_list.append(instruction)
            
    def add_instruction_exec(self, waveform: Waveform) -> None:
        self.add_instruction(EXECInstruction(waveform))
        
    def add_instruction_goto(self, target_block: 'InstructionBlock') -> None:
        self.add_instruction(GOTOInstruction(InstructionPointer(target_block)))
        
    def add_instruction_cjmp(self,
                             trigger: Trigger,
                             target_block: 'InstructionBlock') -> None:
        self.add_instruction(CJMPInstruction(trigger, InstructionPointer(target_block)))
        
    def add_instruction_stop(self) -> None:
        self.add_instruction(STOPInstruction())
      
    @property
    def instructions(self) -> InstructionSequence:
        return self.__instruction_list.copy()

    @property
    def return_ip(self) -> InstructionPointer:
        return self.__return_ip

    @return_ip.setter
    def return_ip(self, value: InstructionPointer) -> None:
        self.__return_ip = value
        
    def create_embedded_block(self) -> 'InstructionBlock':
        block = InstructionBlock()
        self.__embedded_blocks.append(block)
        return block

    def __iter__(self) -> Iterable[Instruction]:
        for instruction in self.__instruction_list.copy():
            yield instruction
        if self.__return_ip is None:
            yield STOPInstruction()
        else:
            yield GOTOInstruction(self.__return_ip)

    def __getitem__(self, index: int) -> Instruction:
        if index < len(self.__instruction_list):
            return self.__instruction_list[index]
        elif index == len(self.__instruction_list):
            if self.__return_ip is None:
                return STOPInstruction()
            else:
                return GOTOInstruction(self.__return_ip)
        else:
            raise IndexError()

    def __len__(self) -> int:
        return len(self.__instruction_list) + 1

    @property
    def compare_key(self) -> Any:
        return id(self)
        
    def __str__(self) -> str:
        return str(hash(self))


class SequenceContinuationStrategy(Comparable, metaclass=ABCMeta):

    def __init__(self) -> None:
        super().__init__()


class WaveformSequence(metaclass=ABCMeta):

    @abstractproperty
    def waveforms(self) -> List[Waveform]:
        pass

    @abstractproperty
    def repetitions(self) -> int:
        pass

    @abstractproperty
    def goto_target(self) -> Optional['WaveformSequence']:
        pass

    @abstractproperty
    def cjmp_trigger(self) -> Optional[Trigger]:
        pass

    @abstractproperty
    def cjmp_target(self) -> Optional['WaveformSequence']:
        pass


class MutableWaveformSequence(WaveformSequence):

    def __init__(self,
                 repetitions: int=1,
                 goto_target: Optional[WaveformSequence]=None,
                 cjmp_trigger: Optional[Trigger]=None,
                 cjmp_target: Optional[WaveformSequence]=None) -> None:
        super().__init__()
        self.__waveforms = []  # type: List[Waveform]
        self.__repetitions = repetitions  # type: int
        self.__goto_target = goto_target  # type: Optional[WaveformSequence]
        self.__cjmp_trigger = cjmp_trigger  # type: Optional[Trigger]
        self.__cjmp_target = cjmp_target  # type: Optional[WaveformSequence]

    def add(self, waveform: Waveform) -> None:
        self.__waveforms.append(waveform)

    @property
    def waveforms(self) -> List[Waveform]:
        return self.__waveforms.copy()

    @property
    def repetitions(self) -> int:
        return self.__repetitions

    @property
    def goto_target(self) -> Optional[WaveformSequence]:
        return self.__goto_target

    @property
    def cjmp_trigger(self) -> Optional[Trigger]:
        return self.__cjmp_trigger

    @property
    def cjmp_target(self) -> Optional[WaveformSequence]:
        return self.__cjmp_target


class ImmutableWaveformSequence(WaveformSequence):

    def __init__(self, sequence: WaveformSequence):
        super().__init__()
        self.__sequence = sequence
        self.__waveforms = self.__sequence.waveforms.copy()

    @property
    def waveforms(self) -> List[Waveform]:
        return self.__waveforms.copy()

    @property
    def repetitions(self) -> int:
        return self.__sequence.repetitions

    @property
    def goto_target(self) -> Optional['ImmutableWaveformSequence']:
        goto_target = self.__sequence.goto_target
        if goto_target is None:
            return None
        return ImmutableWaveformSequence(goto_target)

    @property
    def cjmp_trigger(self) -> Optional[Trigger]:
        return self.__sequence.cjmp_trigger

    @property
    def cjmp_target(self) -> Optional['ImmutableWaveformSequence']:
        cjmp_target = self.__sequence.cjmp_target
        if cjmp_target is None:
            return None
        return ImmutableWaveformSequence(cjmp_target)
