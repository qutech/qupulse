"""STANDARD LIBRARY IMPORTS"""
from abc import ABCMeta, abstractmethod
from typing import List, Tuple

"""RELATED THIRD PARTY IMPORTS"""

"""LOCAL IMPORTS"""

WaveformTableEntry = Tuple[int, float]
WaveformTable = Tuple[WaveformTableEntry, ...]

class Waveform:
    
    def __init__(self, length: int= 0):
        super().__init__()
        if length < 0:
            raise ValueError("length must be a non-negative integer (was {})".format(length))
        self._length = length
        
    def __len__(self):
        return self._length
        
    def __eq__(self, other) -> bool:
        return self is other
        
    def __ne__(self, other) -> bool:
        return not self == other
        
    def __hash__(self) -> int:
        return id(self)
        
    def __str__(self) -> str:
        return str(hash(self))
        
class InstructionPointer:
    
    def __init__(self, block: "InstructionBlock", offset: int):
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
        
    def __hash__(self) -> bool:
        return hash((self.block, self.offset))
        
    def __str__(self) -> str:
        return "IP:{0}#{1}".format(self.block, self.offset)
        
class InstructionBlockAlreadyFinalizedException(Exception):
    """!@brief Indicates that an attempt was made to change an already finalized InstructionBlock."""
    def __str__(self):
        return "An attempt was made to change an already finalized InstructionBlock."
        
        
class InstructionBlockNotYetPlacedException(Exception):
    """!@brief Indicates that an attempt was made to obtain the start address of an InstructionBlock that was not yet placed inside the corresponding outer block."""
    def __str__(self):
        return "An attempt was made to obtain the start address of an InstructionBlock that was not yet placed inside the corresponding outer block."
        
class MissingReturnAddressException(Exception):
    """!@brief Indicates that an inner InstructionBlock has no return address."""
    def __str__(self):
        return "No return address is set!"
        
class InstructionBlock:
    
    def __init__(self, outerBlock: "InstructionBlock" = None):
        super().__init__()
        self._instructionList = [] # type: List[Instruction]
        self._embeddedBlocks = [] # type: List[InstructionBlock]
        self._finalized = False # type: bool
        self._outerBlock = outerBlock
        self._offset = None
        if self._outerBlock is None:
            self._offset = 0
        self.returnIP = None
        
    def _add_instruction(self, instruction: "Instruction") -> None:
        if not self._finalized:
            self._instructionList.append(instruction)
        else:
            raise InstructionBlockAlreadyFinalizedException()
            
    def add_instruction_exec(self, waveformHandle: str) -> None:
        self._add_instruction(EXECInstruction(waveformHandle))
        
    def add_instruction_goto(self, gotoBlock: "InstructionBlock", offset: int = 0) -> None:
        self._add_instruction(GOTOInstruction(gotoBlock, offset))
        
    def add_instruction_cjmp(self, condition) -> "InstructionBlock":
        targetBlock = self._create_embedded_block()
        self._add_instruction(CJMPInstruction(condition, targetBlock, 0))
        targetBlock.returnIP = InstructionPointer(self, len(self))
        return targetBlock
        
    def add_instruction_stop(self) -> None:
        self._add_instruction(STOPInstruction())
        
    def get_instructions(self) -> List["Instruction"]:
        return self._instructionList.copy()
        
    def _create_embedded_block(self) -> "InstructionBlock":
        block = InstructionBlock(self)
        self._embeddedBlocks.append(block)
        return block
        
    def finalize(self) -> None:
        if (self._finalized):
            return
        
        if self._outerBlock is None:
            self.add_instruction_stop()
        elif self.returnIP is not None:
            self.add_instruction_goto(self.returnIP.block, self.returnIP.offset)
        else:
            raise MissingReturnAddressException()
            
        self._finalized = True
            
        for block in self._embeddedBlocks:
            block.finalize()
            block._offset = len(self._instructionList)
            self._instructionList.extend(block._instructionList)
    
    def get_start_address(self) -> int:
        if self._offset is None:
            raise InstructionBlockNotYetPlacedException()
        pos = self._offset
        if self.returnIP is not None:
            pos += self._outerBlock.get_start_address()
        return pos
        
    def is_finalized(self) -> bool:
        return self._finalized
        
    instructions = property(get_instructions)
    finalized = property(is_finalized)
    
    def __len__(self) -> int:
        return len(self._instructionList)
    
    def __eq__(self, other) -> bool:
        return self is other
        
    def __ne__(self, other) -> bool:
        return not self == other
    
    def __hash__(self) -> int:
        return id(self)
        
    def __str__(self) -> str:
        return str(hash(self))
        
class Instruction(metaclass = ABCMeta):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_instruction_code(self) -> str:
        pass
        
    def __str__(self) -> str:
        return self.get_instruction_code()
        
class CJMPInstruction(Instruction):

    def __init__(self, condition, block: InstructionBlock, offset: int = 0):
        super().__init__()
        self.condition = condition
        self.target = InstructionPointer(block, offset)

    def get_instruction_code(self) -> str:
        return 'cjmp'
        
    def __eq__(self, other) -> bool:
        return ((isinstance(other, CJMPInstruction)) and (self.condition == other.condition) and (self.target == other.target))
        
    def __ne__(self, other) -> bool:
        return not self == other
        
    def __hash__(self) -> int:
        return hash((self.condition, self.target))
        
    def __str__(self) -> str:
        return "{0} to {1} on {2}".format(self.get_instruction_code(), self.target, self.condition)
        
class GOTOInstruction(Instruction):
    
    def __init__(self, block: InstructionBlock, offset: int = 0):
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

    def __init__(self, waveform: Waveform):
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

    def __init__(self):
        super().__init__()

    def get_instruction_code(self) -> str:
        return 'stop'
        
    def __eq__(self, other) -> bool:
        return isinstance(other, STOPInstruction)
        
    def __ne__(self, other) -> bool:
        return not self == other
        
    def __hash__(self) -> int:
        return 0