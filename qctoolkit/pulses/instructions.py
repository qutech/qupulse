"""This module defines the abstract hardware instruction model of the qc-toolkit.

Classes:
    - Waveform: An instantiated pulse which can be sampled to a raw voltage value array.
    - Trigger: Representation of a hardware trigger.
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
from typing import List, Any
import numpy

from qctoolkit.comparable import Comparable

# TODO lumip: add docstrings to InstructionBlock after issue #116 is resolved

__all__ = ["Waveform", "Trigger", "InstructionPointer",
           "Instruction", "CJMPInstruction", "EXECInstruction", "GOTOInstruction",
           "STOPInstruction", "InstructionBlock", "InstructionSequence",
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

    @abstractproperty
    def num_channels(self) -> int:
        """The number of channels this waveform is defined for."""


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
    
    def __init__(self, block: 'InstructionBlock', offset: int) -> None:
        super().__init__()
        if offset < 0:
            raise ValueError("offset must be a non-negative integer (was {})".format(offset))
        self.block = block
        self.offset = offset
        
    def get_absolute_address(self) -> int:
        """Return the absolute offset of the targeted instruction in the final instruction sequence.
        """
        return self.block.get_start_address() + self.offset

    @property
    def compare_key(self) -> Any:
        return id(self.block), self.offset
        
    def __str__(self) -> str:
        try:
            return "{}".format(self.get_absolute_address())
        except InstructionBlockNotYetPlacedException:
            return "IP:{0}#{1}".format(self.block, self.offset)


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

    def __init__(self, trigger: Trigger, block: 'InstructionBlock', offset: int=0) -> None:
        super().__init__()
        self.trigger = trigger
        self.target = InstructionPointer(block, offset)

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
    
    def __init__(self, block: 'InstructionBlock', offset: int=0) -> None:
        super().__init__()
        self.target = InstructionPointer(block, offset)

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


class InstructionBlock(Comparable):
    
    def __init__(self, outer_block: 'InstructionBlock'=None) -> None:
        super().__init__()
        self.__instruction_list = [] # type: InstructionSequence
        self.__embedded_blocks = [] # type: List[InstructionBlock]
        self.__outer_block = outer_block
        self.__offset = None
        if self.__outer_block is None:
            self.__offset = 0
        self.return_ip = None
        self.__compiled_sequence = None # type: InstructionSequence
        
    def add_instruction(self, instruction: Instruction) -> None:
        # change to instructions -> invalidate cached compiled sequence
        if self.__compiled_sequence is not None:
            self.__compiled_sequence = None
            for block in self.__embedded_blocks:
                block.__offset = None
        self.__instruction_list.append(instruction)
            
    def add_instruction_exec(self, waveform: Waveform) -> None:
        self.add_instruction(EXECInstruction(waveform))
        
    def add_instruction_goto(self, target_block: 'InstructionBlock', offset: int=0) -> None:
        self.add_instruction(GOTOInstruction(target_block, offset))
        
    def add_instruction_cjmp(self,
                             trigger: Trigger,
                             target_block: 'InstructionBlock',
                             offset: int=0) -> None:
        self.add_instruction(CJMPInstruction(trigger, target_block, offset))
        
    def add_instruction_stop(self) -> None:
        self.add_instruction(STOPInstruction())
      
    @property
    def instructions(self) -> InstructionSequence:
        return self.__instruction_list.copy()
        
    def create_embedded_block(self) -> 'InstructionBlock':
        block = InstructionBlock(self)
        self.__embedded_blocks.append(block)
        return block
        
    def compile_sequence(self) -> InstructionSequence:
        # do not recompile if no changes happened
        if self.__compiled_sequence is not None:
            return self.__compiled_sequence
            
        # clear old offsets
        for block in self.__embedded_blocks:
            block.__offset = None
            
        self.__compiled_sequence = self.__instruction_list.copy()
        
        if self.__outer_block is None:
            self.__compiled_sequence.append(STOPInstruction())
        elif self.return_ip is not None:
            self.__compiled_sequence.append(
                GOTOInstruction(self.return_ip.block, self.return_ip.offset))
        else:
            self.__compiled_sequence = None
            raise MissingReturnAddressException()
            
        for block in self.__embedded_blocks:
            block.__offset = len(self.__compiled_sequence)
            block_sequence = block.compile_sequence()
            self.__compiled_sequence.extend(block_sequence)
            
        return self.__compiled_sequence
    
    def get_start_address(self) -> int:
        if self.__offset is None:
            raise InstructionBlockNotYetPlacedException()
        pos = self.__offset
        if self.__outer_block is not None:
            pos += self.__outer_block.get_start_address()
        return pos
    
    def __len__(self) -> int:
        return len(self.__instruction_list)

    @property
    def compare_key(self) -> Any:
        return id(self)
        
    def __str__(self) -> str:
        return str(hash(self))
