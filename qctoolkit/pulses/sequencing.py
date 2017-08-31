"""This module provides sequencing functionality: It defines classes and algorithms required
to translate PulseTemplates into a hardware understandable abstract instruction sequence of
instantiated pulses or waveforms.

Classes:
    - SequencingElement: Interface for objects that can be translated into instruction sequences.
    - Sequencer: Controller of the sequencing/translation process.
"""

from abc import ABCMeta, abstractmethod
from typing import Tuple, Dict, Union, Optional, List
import numbers

from qctoolkit.utils.types import ChannelID

from . import conditions
from qctoolkit.pulses.instructions import InstructionBlock, ImmutableInstructionBlock, Waveform
from qctoolkit.pulses.parameters import Parameter, ConstantParameter


__all__ = ["SequencingElement", "Sequencer"]


class SequencingElement(metaclass=ABCMeta):
    """An entity which can be sequenced using Sequencer.

    See also:
        Sequencer
    """

    def __init__(self) -> None:
        pass

    def atomicity(self) -> bool:
        """Is the element translated to a single EXECInstruction with one waveform"""
        raise NotImplementedError()

    @abstractmethod
    def build_sequence(self,
                       sequencer: 'Sequencer',
                       parameters: Dict[str, Parameter],
                       conditions: Dict[str, 'conditions.Condition'],
                       measurement_mapping: Dict[str, str],
                       channel_mapping: Dict[ChannelID, ChannelID],
                       instruction_block: InstructionBlock) -> None:
        """Translate this SequencingElement into an instruction sequence for the given
        instruction_block using sequencer and the given parameter and condition sets.

        Implementation guide: Use instruction_block methods to add instructions or create new
        InstructionBlocks. Use sequencer to push child elements to the translation stack.

        Args:
            Sequencer sequencer: The Sequencer object coordinating the current sequencing process.
            Dict[str -> Parameter] parameters: A mapping of parameter names to Parameter objects.
            Dict[str -> Condition] conditions: A mapping of condition identifiers to Condition
                objects.
            Dict[str -> str] measurement_mapping: A mapping of measurement window names
            InstructionBlock instruction_block: The instruction block into which instructions
                resulting from the translation of this SequencingElement will be placed.
        """

    @abstractmethod
    def requires_stop(self,
                      parameters: Dict[str, Parameter],
                      conditions: Dict[str, 'conditions.Condition']) -> bool:
        """Return True if this SequencingElement cannot be translated yet.

        Sequencer will check requires_stop() before calling build_sequence(). If requires_stop()
        returns True, Sequencer interrupts the current translation process and will not call
        build_sequence().

        Implementation guide: requires_stop() should only return True, if this SequencingElement
        cannot be translated, i.e., the return value should only depend on the parameters/conditions
        of this SequencingElement, not on possible child elements.
        If this SequencingElement contains a child element which requires a stop, it should be
        pushed to the sequencing stack nonetheless. The requires_stop information of the child
        will be regarded during translation of that element.

        Args:
            parameters (Dict(str -> Parameter)): A mapping of parameter names to Parameter objects.
            conditions (Dict(str -> Condition)): A mapping of condition identifiers to Condition
                objects.
        Returns:
            True, if this SequencingElement cannot be translated yet. False, otherwise.
        """


class Sequencer:
    """Translates tree structures of SequencingElement objects to linear instruction sequences.

    The concept of the sequencing process itself is described in detail in Section 1.4 of the
    documentation. Sequencer controls the process and invokes the translation functions of
    SequencingElements on the sequencing stack, which in turn implement the details of the
    translation of that specific SequencingElement.

    Sequencer manages a main InstructionBlock into which all instructions are translated by default.
    Since additional InstructionBlocks may be created during the sequencing process due to looping
    and branching, Sequencer maintains several several sequencing stacks - one for each block -
    simultaneously and continues the translation until no stack holds objects that may be
    translated anymore before compiling the final sequence and interrupting or finishing the
    sequencing process.

    See also:
        SequencingElement
    """

    StackElement = Tuple[SequencingElement, Dict[str, Parameter], Dict[str, 'Condition'], Dict[str,str]]

    def __init__(self) -> None:
        """Create a Sequencer."""
        super().__init__()
        self.__waveforms = dict()  # type: Dict[int, Waveform]
        self.__main_block = InstructionBlock()
        self.__sequencing_stacks = \
            {self.__main_block: []}  # type: Dict[InstructionBlock, List[Sequencer.StackElement]]

    def push(self,
             sequencing_element: SequencingElement,
             parameters: Optional[Dict[str, Union[Parameter, float]]]=None,
             conditions: Optional[Dict[str, 'conditions.Condition']]=None,
             *,
             window_mapping: Optional[Dict[str, str]]=None,
             channel_mapping: Optional[Dict[ChannelID, ChannelID]]=None,
             target_block: Optional[InstructionBlock]=None) -> None:
        """Add an element to the translation stack of the target_block with the given set of
         parameters.

        The element will be on top of the stack, i.e., it is the first to be translated if no
        subsequent calls to push with the same target_block occur.

        Args:
            sequencing_element (SequencingElement): The SequencingElement to push to the stack.
            parameters (Dict(str -> (Parameter or float)): A mapping of parameters names defined
                in the SequencingElement to Parameter objects or constant float values. In the
                latter case, the float values are encapsulated into ConstantParameter objects.
                Optional, if no conditions are defined by the SequencingElement. (default: None)
            conditions (Dict(str -> Condition)): A mapping of condition identifier defined by the
                SequencingElement to Condition objects. Optional, if no conditions are defined by
                the SequencingElement. (default: None)
            window_mapping (Dict(str -> str)): Mapping of the measurement window names of the sequence element
            channel_mapping (Dict(ChannelID -> ChannelID)): Mapping of the defined channels
            target_block (InstructionBlock): The instruction block into which instructions resulting
                from the translation of the SequencingElement will be placed. Optional. If not
                provided, the main instruction block will be targeted. (default: None)
        """
        if parameters is None:
            parameters = dict()
        if conditions is None:
            conditions = dict()
        if target_block is None:
            target_block = self.__main_block
        if window_mapping is None:
            if hasattr(sequencing_element, 'measurement_names'):
                window_mapping = {wn: wn for wn in sequencing_element.measurement_names}
            else:
                window_mapping = dict()
        if channel_mapping is None:
            if hasattr(sequencing_element, 'defined_channels'):
                channel_mapping = {cn: cn for cn in sequencing_element.defined_channels}
            else:
                channel_mapping = dict()
        for (key, value) in parameters.items():
            if isinstance(value, numbers.Real):
                parameters[key] = ConstantParameter(value)

        if target_block not in self.__sequencing_stacks:
            self.__sequencing_stacks[target_block] = []

        self.__sequencing_stacks[target_block].append((sequencing_element, parameters, conditions, window_mapping,
                                                       channel_mapping))

    def build(self) -> ImmutableInstructionBlock:
        """Start the translation process. Translate all elements currently on the translation stacks
        into an InstructionBlock hierarchy.

        Processes all sequencing stacks (for each InstructionBlock) until each stack is either
        empty or its topmost element requires a stop. If build is called after a previous
        translation process where some elements required a stop (i.e., has_finished returned False),
        it will append new instructions to the previously generated and returned blocks.

        Returns:
            The instruction block (hierarchy) resulting from the translation of the (remaining)
                SequencingElements on the sequencing stacks.
        """
        if not self.has_finished():
            shall_continue = True # shall_continue will only be False, if the first element on all
                                  # stacks requires a stop or all stacks are empty
            while shall_continue:
                shall_continue = False
                for target_block, sequencing_stack in self.__sequencing_stacks.copy().items():
                    while sequencing_stack:
                        (element, parameters, conditions, window_mapping, channel_mapping) = sequencing_stack[-1]
                        if not element.requires_stop(parameters, conditions):
                            shall_continue |= True
                            sequencing_stack.pop()
                            element.build_sequence(self, parameters, conditions, window_mapping,
                                                   channel_mapping, target_block)
                        else: break

        return ImmutableInstructionBlock(self.__main_block, dict())

    def has_finished(self) -> bool:
        """Check whether all translation stacks are empty. Indicates that the translation is
        complete.

        Note that has_finished will return False, if there are stack elements that require a stop.
        In this case, calling build will only have an effect if these elements no longer require a
        stop, e.g. when required measurement results have been acquired since the last translation.

        Returns:
            Returns True, if all translation stacks are empty, i.e., the translation is complete.
        """
        return not any(self.__sequencing_stacks.values())
