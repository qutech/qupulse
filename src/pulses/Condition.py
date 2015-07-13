"""STANDARD LIBRARY IMPORTS"""
import logging
from abc import ABCMeta, abstractmethod
from typing import Dict, Optional, Callable

"""RELATED THIRD PARTY IMPORTS"""

"""LOCAL IMPORTS"""
from .Parameter import Parameter
from .Sequencer import SequencingElement, Sequencer
from .Instructions import InstructionBlock, InstructionPointer, Trigger

logger = logging.getLogger(__name__)

# TODO lumip: Complete docstrings
# TODO lumip: Tests

class Condition(metaclass = ABCMeta):
    """!@brief A condition on which the execution of a pulse may depend.
    
    Conditions are used for branching and looping of pulses and
    thus relevant for BranchPulseTemplate and LoopPulseTemplate.
    Implementations of Condition may rely on software variables,
    measured data or be mere placeholders for hardware triggers.
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        
    @abstractmethod
    def build_sequence_loop(self, 
            delegator: SequencingElement,
            body: SequencingElement,
            sequencer: Sequencer, 
            time_parameters: Dict[str, Parameter], 
            voltage_parameters: Dict[str, Parameter], 
            instruction_block: InstructionBlock):
        pass
    
    @abstractmethod
    def build_sequence_branch(self,
            delegator: SequencingElement,
            if_branch: SequencingElement, 
            else_branch: SequencingElement, 
            sequencer: Sequencer, 
            time_parameters: Dict[str, Parameter], 
            voltage_parameters: Dict[str, Parameter], 
            instruction_block: InstructionBlock):
        pass


class HardwareCondition(Condition):
    
    def __init__(self, trigger: Trigger):
        super().__init__()
        self.__trigger = trigger # type: Trigger
        

    def build_sequence_loop(self, 
            delegator: SequencingElement,
            body: SequencingElement, 
            sequencer: Sequencer, 
            time_parameters: Dict[str, Parameter], 
            voltage_parameters: Dict[str, Parameter], 
            instruction_block: InstructionBlock):
        
        body_block = instruction_block.create_embedded_block()
        body_block.return_ip = InstructionPointer(instruction_block, len(body_block))
        
        instruction_block.add_instruction_cjmp(self.__trigger, body_block)
        sequencer.push(body, time_parameters, voltage_parameters, body_block)
        
    def build_sequence_branch(self, 
            delegator: SequencingElement,
            if_branch: SequencingElement, 
            else_branch: SequencingElement, 
            sequencer: Sequencer, 
            time_parameters: Dict[str, Parameter], 
            voltage_parameters: Dict[str, Parameter], 
            instruction_block: InstructionBlock):
        
        if_block = instruction_block.create_embedded_block()
        else_block = instruction_block.create_embedded_block()
        
        instruction_block.add_instruction_cjmp(self.__trigger, if_block)
        sequencer.push(if_branch, time_parameters, voltage_parameters, if_block)
        
        instruction_block.add_instruction_goto(else_block)
        sequencer.push(else_branch, time_parameters, voltage_parameters, else_block)
        
        if_block.return_ip = InstructionPointer(instruction_block, len(instruction_block))
        else_block.return_ip = if_block.return_ip

    
class SoftwareCondition(Condition):
        
    def __init__(self, evaluationCallback: Callable[[], Optional[bool]]):
        super().__init__()
        self.__callback = evaluationCallback # type: Callable[[], Optional[bool]]
        

    def build_sequence_loop(self, 
            delegator: SequencingElement,
            body: SequencingElement, 
            sequencer: Sequencer, 
            time_parameters: Dict[str, Parameter], 
            voltage_parameters: Dict[str, Parameter], 
            instruction_block: InstructionBlock):
        
        evaluationResult = self.__callback()
        if evaluationResult is None:
            instruction_block.add_instruction_stop()
            sequencer.push(delegator, time_parameters, voltage_parameters, instruction_block)
        else:
            if evaluationResult == True:
                sequencer.push(delegator, time_parameters, voltage_parameters, instruction_block)
                sequencer.push(body, time_parameters, voltage_parameters, instruction_block)
                
        
    def build_sequence_branch(self, 
            delegator: SequencingElement,
            if_branch: SequencingElement, 
            else_branch: SequencingElement, 
            sequencer: Sequencer, 
            time_parameters: Dict[str, Parameter], 
            voltage_parameters: Dict[str, Parameter], 
            instruction_block: InstructionBlock):
        
        evaluationResult = self.__callback()
        if evaluationResult is None:
            instruction_block.add_instruction_stop()
            sequencer.push(delegator, time_parameters, voltage_parameters, instruction_block)
        else:
            if evaluationResult == True:
                sequencer.push(if_branch, time_parameters, voltage_parameters, instruction_block)
            else:
                sequencer.push(else_branch, time_parameters, voltage_parameters, instruction_block)
                
