"""STANDARD LIBRARY IMPORTS"""
from abc import ABCMeta, abstractmethod
from typing import Tuple, Dict

"""RELATED THIRD PARTY IMPORTS"""

"""LOCAL IMPORTS"""
from .Instructions import InstructionBlock, Waveform, WaveformTable
from .Parameter import Parameter

    
class SequencingElement(metaclass = ABCMeta):
    
    def __init__(self) -> None:
        super().__init__()
        
    @abstractmethod
    def build_sequence(self, sequencer: "Sequencer", time_parameters: Dict[str, Parameter], voltage_parameters: Dict[str, Parameter], instruction_block: InstructionBlock) -> None:
        pass
        
    @abstractmethod
    def requires_stop(self, time_parameters: Dict[str, Parameter], voltage_parameters: Dict[str, Parameter]) -> bool: 
        pass
    
class SequencingHardwareInterface(metaclass = ABCMeta):

    def __init(self) -> None:
        super().__init__()
    
    @abstractmethod
    def register_waveform(self, waveform_table: WaveformTable) -> Waveform:
        pass
    
class Sequencer:

    StackElement = Tuple[SequencingElement, Dict[str, Parameter], Dict[str, Parameter]]

    def __init__(self, hardware_interface: SequencingHardwareInterface) -> None:
        super().__init__()
        self.__hardware_interface = hardware_interface
        self.__waveforms = dict() #type: Dict[int, Waveform]
        self.__main_block = InstructionBlock()
        self.__sequencing_stacks = {self.__main_block: []} #type: Dict[InstructionBlock, List[StackElement]]
        
    def push(self, sequencing_element: SequencingElement, time_parameters: Dict[str, Parameter], voltage_parameters: Dict[str, Parameter], target_block: InstructionBlock = None) -> None:
        if target_block is None:
            target_block = self.__main_block
            
        if not target_block in self.__sequencing_stacks:
            self.__sequencing_stacks[target_block] = []
            
        self.__sequencing_stacks[target_block].append((sequencing_element, time_parameters, voltage_parameters))
        
    def build(self) -> InstructionBlock:
        if not self.has_finished():            
            shall_continue = True # shall_continue will only be False, if the first element on all stacks requires a stop or all stacks are empty
            while shall_continue:
                shall_continue = False
                for target_block, sequencing_stack in self.__sequencing_stacks.copy().items():
                    while sequencing_stack:
                        (element, time_parameters, voltage_parameters) = sequencing_stack[-1]
                        if not element.requires_stop(time_parameters, voltage_parameters):
                            shall_continue |= True
                            sequencing_stack.pop()
                            element.build_sequence(self, time_parameters, voltage_parameters, target_block)
                        else: break
        
        self.__main_block.compile_sequence()
        return self.__main_block
        
    def has_finished(self) -> bool:
        return all(not stack for stack in self.__sequencing_stacks.values())
        
    def register_waveform(self, waveform_table: WaveformTable) -> Waveform:
        waveform_table_hash = hash(waveform_table)
        waveform = None
        if waveform_table_hash in self.__waveforms:
            waveform = self.__waveforms[waveform_table_hash]
        else:
            waveform = self.__hardware_interface.register_waveform(waveform_table)
            self.__waveforms[waveform_table_hash] = waveform
        return waveform
