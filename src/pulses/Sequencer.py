"""STANDARD LIBRARY IMPORTS"""
from abc import ABCMeta, abstractmethod
from typing import Tuple, Dict

"""RELATED THIRD PARTY IMPORTS"""

"""LOCAL IMPORTS"""
from pulses.Instructions import InstructionBlock, Waveform, WaveformTable
from pulses.Parameter import Parameter
    
class SequencingElement(metaclass = ABCMeta):
    
    def __init__(self):
        super().__init__()
        
    @abstractmethod
    def build_sequence(self, sequencer: "Sequencer", timeParameters: Dict[str, Parameter], voltageParameters: Dict[str, Parameter], instructionBlock: InstructionBlock) -> None:
        pass
    
class SequencingHardwareInterface(metaclass = ABCMeta):

    def __init(self):
        super().__init__()
    
    @abstractmethod
    def register_waveform(self, waveformTable: WaveformTable) -> Waveform:
        pass
    
class Sequencer:

    StackElement = Tuple[SequencingElement, Dict[str, Parameter], Dict[str, Parameter], InstructionBlock]

    def __init__(self, hardwareInterface: SequencingHardwareInterface):
        super().__init__()
        self._sequencingStack = [] #type: List[StackElement]
        self._hardwareInterface = hardwareInterface
        self._waveforms = dict() #type: Dict[int, Waveform]
        
    def push(self, pulseTemplate: SequencingElement, timeParameters: Dict[str, Parameter], voltageParameters: Dict[str, Parameter], targetBlock: InstructionBlock = None) -> None:
        self._sequencingStack.append((pulseTemplate, timeParameters, voltageParameters, targetBlock))
        
    def build(self) -> InstructionBlock:
        mainBlock = InstructionBlock()
        if not self.has_finished():
            (template, timeParameters, voltageParameters, targetBlock) = self._sequencingStack.pop()
            while True: # there seems to be no do-while loop in python and "while" True with a break condition is a suggested solution
                if targetBlock is None:
                    targetBlock = mainBlock
                template.build_sequence(self, timeParameters, voltageParameters, targetBlock)
                if (self.has_finished()) or (self._sequencingStack[-1].requires_stop()):
                    break
                (template, timeParameters, voltageParameters, targetBlock) = self._sequencingStack.pop()
        
        mainBlock.finalize()
        return mainBlock
        
    def has_finished(self):
        return not self._sequencingStack
        
    def register_waveform(self, waveformTable: WaveformTable) -> Waveform:
        waveformTableHash = hash(waveformTable)
        waveform = None
        if waveformTableHash in self._waveforms:
            waveform = self._waveforms[waveformTableHash]
        else:
            waveform = self._hardwareInterface.register_waveform(waveformTable)
            self._waveforms[waveformTableHash] = waveform
        return waveform
