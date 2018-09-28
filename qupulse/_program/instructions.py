"""This module defines the abstract hardware instruction model of qupulse.

Classes:
    - Trigger: Representation of a hardware trigger.
    - Instruction: Base class for hardware instructions.
    - CJMPInstruction: Conditional jump instruction.
    - REPJInstruction: Repetition jump instruciton.
    - EXECInstruction: Instruction to execute a waveform.
    - GOTOInstruction: Unconditional jump instruction.
    - STOPInstruction: Instruction which indicates the end of execution.
    - AbstractInstructionBlock: A block of instructions (abstract base class)
    - InstructionBlock: A mutable block of instructions to which instructions can be added
    - ImmutableInstructionBlock: An immutable InstructionBlock
    - InstructionSequence: A single final sequence of instructions.
    - InstructionPointer: References an instruction's location in a sequence.
"""

from abc import ABCMeta, abstractmethod
from typing import List, Any, Dict, Iterable, Optional, Sequence, Union, Set, Tuple
from weakref import WeakValueDictionary

import numpy

from qupulse.utils.types import ChannelID, MeasurementWindow, TimeType
from qupulse.comparable import Comparable
from qupulse._program.waveforms import Waveform

__all__ = ["Trigger",
           "InstructionPointer", "Instruction", "CJMPInstruction", "EXECInstruction",
           "GOTOInstruction", "STOPInstruction", "REPJInstruction", "AbstractInstructionBlock", "InstructionBlock",
           "ImmutableInstructionBlock", "InstructionSequence", "ChannelID"]


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
    """Reference to the location of an instruction used in expressing targets of jumps.

    The target instruction is referenced by the instruction block it resides in and its offset
    within this block.
    """
    
    def __init__(self, block: 'AbstractInstructionBlock', offset: int=0) -> None:
        """Create a new InstructionPointer instance.

        Args:
            block: The instruction block the referenced instruction
                resides in.
            offset: The position/offset of the referenced instruction in its block.
        Raises:
            ValueError: If offset is negative
        """
        super().__init__()
        if offset < 0:
            raise ValueError("offset must be a non-negative integer (was {})".format(offset))
        self.__block = block
        self.__offset = offset

    @property
    def block(self) -> 'AbstractInstructionBlock':
        """The instruction block containing the referenced instruction."""
        return self.__block

    @property
    def offset(self) -> int:
        """The offset of the referenced instruction in its containing block."""
        return self.__offset

    @property
    def compare_key(self) -> Any:
        return id(self.__block), self.__offset
        
    def __str__(self) -> str:
        return "IP:{0}#{1}".format(self.__block, self.__offset)


class Instruction(Comparable, metaclass=ABCMeta):
    """A hardware instruction."""

    def __init__(self) -> None:
        super().__init__()


class CJMPInstruction(Instruction):
    """A conditional jump hardware instruction.

    Will cause the execution to jump to the instruction indicated by the InstructionPointer held
    by this CJMPInstruction if the given Trigger was fired. If not, this Instruction will have no
    effect, the execution will continue with the following.
    """

    def __init__(self, trigger: Trigger, target: InstructionPointer) -> None:
        """Create a new CJMPInstruction object.

        Args:
            trigger (Trigger): Representation of the hardware trigger which controls whether the
                conditional jump occurs or not.
            target (InstructionPointer): Instruction pointer referencing the instruction targeted
                by the conditional jump.
        """
        super().__init__()
        self.trigger = trigger
        self.target = target

    @property
    def compare_key(self) -> Any:
        return self.trigger, self.target
        
    def __str__(self) -> str:
        return "cjmp to {} on {}".format(self.target, self.trigger)


class MEASInstruction(Instruction):
    """A measurement instruction.

    Cause a measurement to be executed. The instruction itself takes no time."""
    def __init__(self, measurements: List[MeasurementWindow]):
        super().__init__()

        self.measurements = measurements

    @property
    def compare_key(self) -> List[MeasurementWindow]:
        return self.measurements

    def __str__(self):
        return "meas [" + " ,".join(set(name for name, *_ in self.measurements)) + ']'


class REPJInstruction(Instruction):
    """A repetition jump instruction.

    Will cause the execution to jump to the instruction indicated by the InstructionPointer held by
    this REPJInstruction for the first n times this REPJInstruction is encountered, where n is
    a parameter."""

    def __init__(self, count: int, target: InstructionPointer) -> None:
        """Create a new REPJInstruction object.

        Args:
            count (int): A positive integer indicating how often the repetition jump is triggered.
            target (InstructionPointer): Instruction pointer referencing the instruction targeted
                by the repetition jump.
        Raises:
            ValueError, if count is a negative number.
        """
        super().__init__()
        if count < 0:
            raise ValueError("Repetition count must not be negative.")
        self.count = count
        self.target = target

    @property
    def compare_key(self) -> Any:
        return self.count, self.target

    def __str__(self) -> str:
        return "repj {} times to {}".format(self.count, self.target)


class GOTOInstruction(Instruction):
    """An unconditional jump hardware instruction.

    Will cause the execution to jump to the instruction indicated by the InstructionPointer
    held by this GOTOInstruction.
    """
    
    def __init__(self, target: InstructionPointer) -> None:
        """Create a new GOTOInstruction object.

        Args:
            target (InstructionPointer): Instruction pointer referencing the instruction targeted
                by the unconditional jump.
        """
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
        """Create a new EXECInstruction object.

        Args:
            waveform: The waveform that will be executed by this instruction.
        """
        super().__init__()
        self.waveform = waveform

    @property
    def compare_key(self) -> Waveform:
        return self.waveform

    def __str__(self) -> str:
        return "exec {}".format(self.waveform)


class STOPInstruction(Instruction):
    """An instruction which indicates the end of the program."""

    def __init__(self) -> None:
        """Create a new STOPInstruction object."""
        super().__init__()

    @property
    def compare_key(self) -> Any:
        return 0

    def __str__(self) -> str:
        return "stop"


class CHANInstruction(Instruction):
    """Split the control flow for different channels.

    There is no guarantee at this point that the instruction blocks have the same length. This is basically a
    switch statement.
    """

    def __init__(self, channel_to_instruction_block: Dict[ChannelID, InstructionPointer]):
        self.channel_to_instruction_block = channel_to_instruction_block

    @property
    def compare_key(self) -> Dict[ChannelID, InstructionPointer]:
        return self.channel_to_instruction_block

    def __str__(self) -> str:
        return "chan " + ", ".join("{target} for {channel}"
                                  .format(target=v, channel=k)
                                   for k, v in sorted(self.channel_to_instruction_block.items(), key=lambda arg: arg[0]))

    def __getitem__(self, item) -> InstructionPointer:
        return self.channel_to_instruction_block[item]

        
InstructionSequence = List[Instruction] # pylint: disable=invalid-name,invalid-sequence-index


class AbstractInstructionBlock(Comparable, metaclass=ABCMeta):
    """"Abstract base class of a block of instructions representing a (sub)sequence in the control
    flow of a pulse template instantiation.

    Because of included jump instructions, instruction blocks typically form a "calling" hierarchy.
    Due to how the sequencing process works, this hierarchy will typically resemble the pulse
    template from which it was translated closely.

    An instruction block might define a return instruction pointer specifying to which instruction
    the control flow should return after execution of the block has finished.

    Instruction blocks define the item access and the iterable interface to allow access to the
    contained instructions. When using these interfaces, a final stop or goto instruction  is
    automatically added after the regular instructions according to whether a return instruction
    pointer was set or not (to return control flow to a calling block or stop the execution).
    Consequently, the len() operation includes this additional instruction in the returned length.

    The property "instructions" allows access to the contained instructions without the
    additional stop/goto instruction mentioned above.

    See Also:
        InstructionBlock
        ImmutableInstructionBlock
    """

    def __init__(self) -> None:
        """Create a new AbstractInstructionBlock instance."""
        super().__init__()

    @property
    @abstractmethod
    def instructions(self) -> Sequence[Instruction]:
        """The instructions contained in this block (excluding a final stop or return goto)."""

    @property
    @abstractmethod
    def return_ip(self) -> Optional[InstructionPointer]:
        """The return instruction pointer indicating the instruction to which the control flow
        shall return after exection of this instruction block has finished."""

    @property
    def compare_key(self) -> Any:
        return id(self)

    def __str__(self) -> str:
        return str(hash(self))

    def __iter__(self) -> Iterable[Instruction]:
        for instruction in self.instructions:
            yield instruction
        if self.return_ip is None:
            yield STOPInstruction()
        else:
            yield GOTOInstruction(self.return_ip)

    def __getitem__(self, index: Union[int,slice]) -> Union[Instruction,Iterable[Instruction]]:
        if isinstance(index, slice):
            return (self[i] for i in range(*index.indices(len(self))))

        if index > len(self.instructions) or index < -(len(self.instructions) + 1):
            raise IndexError()
        elif index < 0:
            return self[len(self) + index]
        elif index < len(self.instructions):
            return self.instructions[index]
        else:
            if self.return_ip is None:
                return STOPInstruction()
            else:
                return GOTOInstruction(self.return_ip)

    def __len__(self) -> int:
        return len(self.instructions) + 1


class InstructionBlock(AbstractInstructionBlock):
    """A block of instructions representing a (sub)sequence in the control
    flow of a pulse template instantiation.

    Because of included jump instructions, instruction blocks typically form a "calling" hierarchy.
    Due to how the sequencing process works, this hierarchy will typically resemble the pulse
    template from which it was translated closely.

    An instruction block might define a return instruction pointer specifying to which instruction
    the control flow should return after execution of the block has finished.

    Instruction blocks define the item access and the iterable interface to allow access to the
    contained instructions. When using these interfaces, a final stop or goto instruction  is
    automatically added after the regular instructions according to whether a return instruction
    pointer was set or not (to return control flow to a calling block or stop the execution).
    Consequently, the len() operation includes this additional instruction in the returned length.

    The property "instructions" allows access to the contained instructions without the
    additional stop/goto instruction mentioned above."""

    def __init__(self) -> None:
        """Create a new InstructionBlock instance."""
        super().__init__()
        self.__instruction_list = [] # type: InstructionSequence

        self.__return_ip = None

    def add_instruction(self, instruction: Instruction) -> None:
        """Append an instruction at the end of this instruction block.

        Args:
            instruction (Instruction): The instruction to append.
        """
        self.__instruction_list.append(instruction)

    def add_instruction_exec(self, waveform: Waveform) -> None:
        """Create and append a new EXECInstruction object for the given waveform at the end of this
        instruction block.

        Args:
            waveform (Waveform): The Waveform object referenced by the new EXECInstruction.
        """
        self.add_instruction(EXECInstruction(waveform))

    def add_instruction_goto(self, target_block: 'InstructionBlock') -> None:
        """Create and append a new GOTOInstruction object with a given target block at the end of
        this instruction block.

        Args:
            target_block (InstructionBlock): The instruction block the new GOTOInstruction will
                jump to. Execution will begin at the start of that block, i.e., the offset of the
                instruction pointer of the GOTOInstruction will be zero.
        """
        self.add_instruction(GOTOInstruction(InstructionPointer(target_block)))

    def add_instruction_cjmp(self,
                             trigger: Trigger,
                             target_block: 'InstructionBlock') -> None:
        """Create and append a new CJMPInstruction object at the end of this instruction block.

        Args:
            trigger (Trigger): The hardware trigger that will control the new CJMPInstruction.
            target_block (InstructionBlock): The instruction block the new CJMPInstruction will
                jump to. Execution will begin at the start of that block, i.e., the offset of the
                instruction pointer of the CJMPInstruction will be zero.
        """
        self.add_instruction(CJMPInstruction(trigger, InstructionPointer(target_block)))

    def add_instruction_repj(self,
                             count: int,
                             target_block: 'InstructionBlock') -> None:
        """Create and append a new REPJInstruction object at the end of this instruction block.

        Args:
            count (int): The amount of repetitions of the new REPJInstruction.
            target_block (InstructionBlock): The instruction block the new REPJInstruction will
                jump to. Execution will begin at the start of that block, i.e., the offset of the
                instruction pointer of the REPJInstruction will be zero.
        """
        self.add_instruction(REPJInstruction(count, InstructionPointer(target_block)))
        
    def add_instruction_stop(self) -> None:
        """Create and append a new STOPInstruction object at the end of this instruction block."""
        self.add_instruction(STOPInstruction())

    def add_instruction_chan(self, channel_to_instruction: Dict[ChannelID, 'InstructionBlock']) -> None:
        """Create and append a new CHANInstruction at the end of this instruction block."""
        self.add_instruction(CHANInstruction({ch: InstructionPointer(block) for ch, block in channel_to_instruction.items()}))

    def add_instruction_meas(self, measurements: List['MeasurementWindow']):
        """Create and append a MEASInstruction at the end of the instruction block.

        :param measurements: The measurement windows this instruction causes
        """
        self.add_instruction(MEASInstruction(measurements=measurements))

    @property
    def instructions(self) -> InstructionSequence:
        return self.__instruction_list.copy()

    @property
    def return_ip(self) -> InstructionPointer:
        return self.__return_ip

    @return_ip.setter
    def return_ip(self, value: InstructionPointer) -> None:
        self.__return_ip = value

    def __len__(self) -> int:
        return len(self.__instruction_list) + 1


class ImmutableInstructionBlock(AbstractInstructionBlock):
    """An immutable instruction block which cannot be altered.

    See Also:
        InstructionBlock
    """

    def __init__(self,
                 block: AbstractInstructionBlock,
                 context: Dict[AbstractInstructionBlock, 'ImmutableInstructionBlock']=None) -> None:
        """Create a new ImmutableInstructionBlock hierarchy from a (mutable) InstructionBlock
        hierarchy.

        Will create a deep copy (including all embedded blocks) of the given instruction block.

        Args:
            block (AbstractInstructionBlock): The instruction block that will be copied into an
                immutable one.
            context (Dict(AbstractInstructionBlock -> ImmutableInstructionBlock)): A dictionary
                to look up already existing conversions of instruction blocks. Required to resolve
                return instruction pointers. Will be altered by the process.
        """
        super().__init__()
        if context is None:
            context = dict()
        self.__return_ip = None
        return_ip = block.return_ip
        if return_ip is not None:
            self.__return_ip = InstructionPointer(context[return_ip.block], return_ip.offset)
        context[block] = self

        def make_immutable(instruction: Instruction) -> Instruction:
            if isinstance(instruction, GOTOInstruction):
                return GOTOInstruction(
                    InstructionPointer(
                        ImmutableInstructionBlock(instruction.target.block, context),
                        instruction.target.offset)
                )
            elif isinstance(instruction, REPJInstruction):
                return REPJInstruction(
                    instruction.count,
                    InstructionPointer(
                        ImmutableInstructionBlock(instruction.target.block, context),
                        instruction.target.offset)
                )
            elif isinstance(instruction, CJMPInstruction):
                return CJMPInstruction(
                    instruction.trigger,
                    InstructionPointer(
                        ImmutableInstructionBlock(instruction.target.block, context),
                        instruction.target.offset)
                )
            else:
                return instruction

        self._instruction_tuple = tuple(make_immutable(instr) for instr in block.instructions)

    @property
    def instructions(self) -> Tuple[Instruction, ...]:
        return self._instruction_tuple

    @property
    def return_ip(self) -> InstructionPointer:
        return self.__return_ip


