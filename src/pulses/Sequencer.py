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

    StackElement = Tuple[SequencingElement, Dict[str, Parameter], Dict[str, Parameter], InstructionBlock]

    def __init__(self, hardware_interface: SequencingHardwareInterface) -> None:
        super().__init__()
        self.__sequencing_stack = [] #type: List[StackElement]
        self.__hardware_interface = hardware_interface
        self.__waveforms = dict() #type: Dict[int, Waveform]
        
    def push(self, sequencing_element: SequencingElement, time_parameters: Dict[str, Parameter], voltage_parameters: Dict[str, Parameter], target_block: InstructionBlock = None) -> None:
        self.__sequencing_stack.append((sequencing_element, time_parameters, voltage_parameters, target_block))
        
    def build(self) -> InstructionBlock:
        main_block = InstructionBlock()
        if not self.has_finished():
            (element, time_parameters, voltage_parameters, target_block) = self.__sequencing_stack.pop()
            while True: # there seems to be no do-while loop in python and "while True" with a break condition is a suggested solution
                if target_block is None:
                    target_block = main_block
                element.build_sequence(self, time_parameters, voltage_parameters, target_block)
                if (self.has_finished()) or (self.__sequencing_stack[-1][0].requires_stop(time_parameters, voltage_parameters)):
                    break
                (element, time_parameters, voltage_parameters, target_block) = self.__sequencing_stack.pop()
        
        main_block.compile_sequence()
        return main_block
        
    def has_finished(self) -> bool:
        return not self.__sequencing_stack
        
    def register_waveform(self, waveform_table: WaveformTable) -> Waveform:
        waveform_table_hash = hash(waveform_table)
        waveform = None
        if waveform_table_hash in self.__waveforms:
            waveform = self.__waveforms[waveform_table_hash]
        else:
            waveform = self.__hardware_interface.register_waveform(waveform_table)
            self.__waveforms[waveform_table_hash] = waveform
        return waveform
