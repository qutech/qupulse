"""STANDARD LIBRARY IMPORTS"""
from abc import ABCMeta, abstractmethod
from typing import List, Tuple

"""RELATED THIRD PARTY IMPORTS"""

"""LOCAL IMPORTS"""

# TODO lumip: add docstrings


__all__ = ['WaveformTableEntry', 'WaveformTable', 'Waveform', 'Trigger', 'InstructionPointer', 'InstructionSequence',
            'InstructionBlockNotYetPlacedException', 'MissingReturnAddressException', 'InstructionBlock',
            'Instruction', 'EXECInstruction', 'CJMPInstruction', 'GOTOInstruction', 'STOPInstruction'
          ]

WaveformTableEntry = Tuple[int, float]
WaveformTable = Tuple[WaveformTableEntry, ...]

class Waveform:
    
    def __init__(self, length: float = 0) -> None:
        super().__init__()
        if length < 0:
            raise ValueError("length must be a non-negative integer (was {})".format(length))
        self.__length = length
        
    def __len__(self) -> int:
        return self.__length
        
    def __eq__(self, other) -> bool:
        return self is other
        
    def __ne__(self, other) -> bool:
        return not self == other
        
    def __hash__(self) -> int:
        return id(self)
        
    def __str__(self) -> str:
        return str(hash(self))
    
class Trigger:
        
    def __init__(self) -> None:
        super().__init__()
        
    def __eq__(self, other) -> bool:
        return self is other
    
    def __ne__(self, other) -> bool:
        return not self == other
    
    def __hash__(self) -> int:
        return id(self)
    
    def __str__(self) -> str:
        return "Trigger {}".format(hash(self))
        
class InstructionPointer:
    
    def __init__(self, block: 'InstructionBlock', offset: int) -> None:
        super().__init__()
        if offset < 0:
            raise ValueError("offset must be a non-negative integer (was {})".format(offset))
        self.block = block
        self.offset = offset
        
    def get_absolute_address(self) -> int:
        return self.block.get_start_address() + self.offset
        
    def __eq__(self, other) -> bool:
        return (isinstance(other, InstructionPointer)) and (self.block is other.block) and (self.offset == other.offset)
        
    def __ne__(self, other) -> bool:
        return not self == other
        
    def __hash__(self) -> int:
        return hash((self.block, self.offset))
        
    def __str__(self) -> str:
        try:
            return "{}".format(self.get_absolute_address())
        finally:
            return "IP:{0}#{1}".format(self.block, self.offset)
        
class Instruction(metaclass = ABCMeta):

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def get_instruction_code(self) -> str:
        pass
        
    def __str__(self) -> str:
        return self.get_instruction_code()
        
class CJMPInstruction(Instruction):

    def __init__(self, trigger: Trigger, block: 'InstructionBlock', offset: int = 0) -> None:
        super().__init__()
        self.trigger = trigger
        self.target = InstructionPointer(block, offset)

    def get_instruction_code(self) -> str:
        return 'cjmp'
        
    def __eq__(self, other) -> bool:
        return (isinstance(other, CJMPInstruction)) and (self.trigger == other.trigger) and (self.target == other.target)
        
    def __ne__(self, other) -> bool:
        return not self == other
        
    def __hash__(self) -> int:
        return hash((self.trigger, self.target))
        
    def __str__(self) -> str:
        return "{0} to {1} on {2}".format(self.get_instruction_code(), self.target, self.trigger)
        
class GOTOInstruction(Instruction):
    
    def __init__(self, block: 'InstructionBlock', offset: int = 0) -> None:
        super().__init__()
        self.target = InstructionPointer(block, offset)
        
    def get_instruction_code(self) -> str:
        return 'goto'
        
    def __eq__(self, other) -> bool:
        return (isinstance(other, GOTOInstruction)) and (self.target == other.target)
        
    def __ne__(self, other) -> bool:
        return not self == other
        
    def __hash__(self) -> int:
        return hash(self.target)
        
    def __str__(self) -> str:
        return "{0} to {1}".format(self.get_instruction_code(), self.target)
        
class EXECInstruction(Instruction):

    def __init__(self, waveform: Waveform) -> None:
        super().__init__()
        self.waveform = waveform
        
    def get_instruction_code(self) -> str:
        return 'exec'
        
    def __eq__(self, other) -> bool:
        return (isinstance(other, EXECInstruction)) and (self.waveform == other.waveform)
        
    def __ne__(self, other) -> bool:
        return not self == other
        
    def __hash__(self) -> int:
        return hash(self.waveform)
        
    def __str__(self) -> str:
        return "{0} {1}".format(self.get_instruction_code(), self.waveform)
        
class STOPInstruction(Instruction):

    def __init__(self) -> None:
        super().__init__()

    def get_instruction_code(self) -> str:
        return 'stop'
        
    def __eq__(self, other) -> bool:
        return isinstance(other, STOPInstruction)
        
    def __ne__(self, other) -> bool:
        return not self == other
        
    def __hash__(self) -> int:
        return 0
        
        
class InstructionBlockAlreadyFinalizedException(Exception):
    """Indicates that an attempt was made to change an already finalized InstructionBlock."""
    def __str__(self) -> str:
        return "An attempt was made to change an already finalized InstructionBlock."
        
        
class InstructionBlockNotYetPlacedException(Exception):
    """Indicates that an attempt was made to obtain the start address of an InstructionBlock that was not yet placed inside the corresponding outer block."""
    def __str__(self) -> str:
        return "An attempt was made to obtain the start address of an InstructionBlock that was not yet finally placed inside the corresponding outer block."
        
class MissingReturnAddressException(Exception):
    """Indicates that an inner InstructionBlock has no return address."""
    def __str__(self) -> str:
        return "No return address is set!"
        
        
InstructionSequence = List[Instruction]
        
class InstructionBlock:
    
    def __init__(self, outerBlock: 'InstructionBlock' = None) -> None:
        super().__init__()
        self.__instruction_list = [] # type: InstructionSequence
        self.__embedded_blocks = [] # type: Collection[InstructionBlock]
        self.__outer_block = outerBlock
        self.__offset = None
        if self.__outer_block is None:
            self.__offset = 0
        self.return_ip = None
        self.__compiled_sequence = None # type: InstructionSequence
        
    def __add_instruction(self, instruction: Instruction) -> None:
        # change to instructions -> invalidate cached compiled sequence
        if self.__compiled_sequence is not None:
            self.__compiled_sequence = None
            for block in self.__embedded_blocks:
                block.__offset = None 
        self.__instruction_list.append(instruction)
            
    def add_instruction_exec(self, waveform: Waveform) -> None:
        self.__add_instruction(EXECInstruction(waveform))
        
    def add_instruction_goto(self, target_block: 'InstructionBlock', offset: int = 0) -> None:
        self.__add_instruction(GOTOInstruction(target_block, offset))
        
    def add_instruction_cjmp(self, trigger: Trigger, target_block: 'InstructionBlock', offset: int = 0) -> None:
        self.__add_instruction(CJMPInstruction(trigger, target_block, offset))
        
    def add_instruction_stop(self) -> None:
        self.__add_instruction(STOPInstruction())
      
    @property
    def instructions(self) -> InstructionSequence:
        return self.__instruction_list.copy()
        
    def create_embedded_block(self) -> 'InstructionBlock':
        block = InstructionBlock(self)
        self.__embedded_blocks.append(block)
        return block
        
    def __get_sequence_length(self) -> int:
        sequence_length = len(self) + 1
        for block in self.__embedded_blocks:
            sequence_length += block.__get_sequence_length()
        return sequence_length        
        
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
            self.__compiled_sequence.append(GOTOInstruction(self.return_ip.block, self.return_ip.offset))
        else:
            self.__compiled_sequence = None
            raise MissingReturnAddressException()
            
        for block in self.__embedded_blocks:
            block.__offset = len(self.__compiled_sequence)
            blockSequence = block.compile_sequence()
            self.__compiled_sequence.extend(blockSequence)
            
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
    
    def __eq__(self, other) -> bool:
        return self is other
        
    def __ne__(self, other) -> bool:
        return not self == other
    
    def __hash__(self) -> int:
        return id(self)
        
    def __str__(self) -> str:
        return str(hash(self))