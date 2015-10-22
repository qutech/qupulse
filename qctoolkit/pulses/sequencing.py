from abc import ABCMeta, abstractmethod, abstractproperty
from typing import Tuple, Dict, Union
import numbers

"""RELATED THIRD PARTY IMPORTS"""

"""LOCAL IMPORTS"""
from .instructions import InstructionBlock, Waveform
from .parameters import Parameter, ConstantParameter


__all__ = ["SequencingElement", "Sequencer", "SequencingHardwareInterface"]

    
class SequencingElement(metaclass = ABCMeta):
    """An entity which can be sequenced using Sequencer."""
    
    def __init__(self) -> None:
        super().__init__()
        
    @abstractmethod
    def build_sequence(self, sequencer: "Sequencer", parameters: Dict[str, Parameter], instruction_block: InstructionBlock) -> None:
        """Translate this SequencingElement into an instruction sequence for the given instruction_block
        using sequencer and the given parameter sets.
        
        Implementation guide: Use instruction_block methods to add instructions or create new InstructionBlocks.
        Use sequencer to push child elements to the translation stack.
        """
        
    @abstractmethod
    def requires_stop(self, parameters: Dict[str, Parameter]) -> bool: 
        """Return True if this SequencingElement cannot be translated yet.
        
        Sequencer will check requires_stop() before calling build_sequence(). If requires_stop() returns True,
        Sequencer interrupts the current translation process and will not call build_sequence().
        
        Implementation guide: requires_stop() should only return True, if this SequencingElement cannot be build,
        i.e., the return value should only depend on the parameters/conditions of this SequencingElement, not on
        possible child elements.
        If this SequencingElement contains a child element which requires a stop, this information will be
        regarded during translation of that element.
        """ 


class SequencingHardwareInterface(metaclass = ABCMeta):
    """A hardware interface used by Sequencer to register waveforms."""

    def __init__(self) -> None:
        super().__init__()
    
    @abstractmethod
    def register_waveform(self, waveform: Waveform) -> None:
        """Register a waveform with the device.
        
        The waveform is given as a waveform_table of (time, voltage) tuples of type (int >= 0, float).
        register_waveform converts this into a sequence of voltages per tick (depending on the hardware
        output frequency) and for this purpose interpolates between entries.
        The function ensures that the resulting waveform is known by the device and returns a Waveform
        object representing a handle to the waveform on the device.
        """


class Sequencer:
    """Translates tree structures of SequencingElement objects to linear instruction sequences contained
    in a InstructionBlock.
    """

    StackElement = Tuple[SequencingElement, Dict[str, Parameter]]

    def __init__(self, hardware_interface: SequencingHardwareInterface) -> None:
        super().__init__()
        self.__hardware_interface = hardware_interface
        self.__waveforms = dict() #type: Dict[int, Waveform]
        self.__main_block = InstructionBlock()
        self.__sequencing_stacks = {self.__main_block: []} #type: Dict[InstructionBlock, List[StackElement]]
        
    def push(self, sequencing_element: SequencingElement, parameters: Dict[str, Union[Parameter, float]], target_block: InstructionBlock = None) -> None:
        """Add an element to the translation stack of the target_block with the given set of parameters.
        
        The element will be on top of the stack, i.e., it is the first to be translated if no subsequent calls
        to push with the same target_block occur.
        """
        if target_block is None:
            target_block = self.__main_block

        for (key, value) in parameters.items():
            if isinstance(value, numbers.Real):
                parameters[key] = ConstantParameter(value)
            
        if target_block not in self.__sequencing_stacks:
            self.__sequencing_stacks[target_block] = []
            
        self.__sequencing_stacks[target_block].append((sequencing_element, parameters))
        
    def build(self) -> InstructionBlock:
        """Start the translation process. Translate all elements currently on the translation stacks into a sequence
         and return the InstructionBlock of the main sequence.
        
        Processes all stacks (for each InstructionBlock) until each stack is either empty or its topmost element
        requires a stop. If build is called after a previous translation process where some elements required a stop
        (i.e., has_finished returned False), it will append new modify the previously generated and returned main
        InstructionBlock. Make sure not to rely on that being unchanged.
        """
        if not self.has_finished():            
            shall_continue = True # shall_continue will only be False, if the first element on all stacks requires a stop or all stacks are empty
            while shall_continue:
                shall_continue = False
                for target_block, sequencing_stack in self.__sequencing_stacks.copy().items():
                    while sequencing_stack:
                        (element, parameters) = sequencing_stack[-1]
                        if not element.requires_stop(parameters):
                            shall_continue |= True
                            sequencing_stack.pop()
                            element.build_sequence(self, parameters, target_block)
                        else: break
        
        self.__main_block.compile_sequence()
        return self.__main_block
        
    def has_finished(self) -> bool:
        """Returns True, if all translation stacks are empty. Indicates that the translation is complete.
        
        Note that has_finished that has_finished will return False, if there are stack elements that require a stop.
        In this case, calling build will only have an effect if these elements no longer require a stop,
        e.g. when required measurement results have been acquired since the last translation.
        """
        return not any(self.__sequencing_stacks.values())

    def register_waveform(self, waveform: Waveform) -> None:
        """Register a waveform with the sequencer.
        
        Forwards waveform registration to the actual hardware (using its SequencingHardwareInterface instance)
        but ensures that identical waveforms are only registered once by maintaining a hash table of previously
        registered waveforms.
        """
        waveform_hash = hash(waveform)
        if waveform_hash not in self.__waveforms:
            self.__hardware_interface.register_waveform(waveform)
            self.__waveforms[waveform_hash] = waveform
