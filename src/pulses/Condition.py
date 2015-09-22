
import logging
from abc import ABCMeta, abstractmethod
from typing import Dict, Optional, Callable

"""RELATED THIRD PARTY IMPORTS"""

"""LOCAL IMPORTS"""
from .Parameter import Parameter
from .Sequencer import SequencingElement, Sequencer
from .Instructions import InstructionBlock, InstructionPointer, Trigger

logger = logging.getLogger(__name__)

class Condition(metaclass = ABCMeta):
    """A condition on which the execution of a pulse may depend.
    
    Conditions are used for branching and looping of pulses and
    thus relevant for BranchPulseTemplate and LoopPulseTemplate.
    Implementations of Condition may rely on software variables,
    measured data or be mere placeholders for hardware triggers.
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        
    @abstractmethod
    def requires_stop(self) -> bool:
        """Return True if evaluating this Condition is not possible in the current translation process."""
        pass
        
    @abstractmethod
    def build_sequence_loop(self, 
                            delegator: SequencingElement,
                            body: SequencingElement,
                            sequencer: Sequencer,
                            parameters: Dict[str, Parameter],
                            instruction_block: InstructionBlock) -> None:
        """Translate a looping SequencingElement using this Condition into an instruction sequence for the given instruction block using sequencer and the given parameter sets.
        
        delegator refers to the SequencingElement which has delegated the invocation of build_sequence to this Condition object. body is the loop body element.
        See also SequencingElement.build_sequence().
        """
        pass
    
    @abstractmethod
    def build_sequence_branch(self,
                              delegator: SequencingElement,
                              if_branch: SequencingElement,
                              else_branch: SequencingElement,
                              sequencer: Sequencer,
                              parameters: Dict[str, Parameter],
                              instruction_block: InstructionBlock) -> None:
        """Translate a branching SequencingElement using this Condition into an instruction sequence for the given instruction block using sequencer and the given parameter sets.
        
        delegator refers to the SequencingElement which has delegated the invocation of build_sequence to this Condition object. if_branch and else_branch are the elements to
        be translated into if and else branch instructions.
        See also SequencingElement.build_sequence().
        """
        pass


class HardwareCondition(Condition):
    """A condition that will be evaluated using hardware triggers.
    
    During the translation process, HardwareCondition instanced will produce in code blocks for branches/loop bodies and the corresponding conditional jump instructions.
    """
    
    def __init__(self, trigger: Trigger) -> None:
        """Create a new HardwareCondition instance. Argument trigger is the trigger handle of the corresponding hardware device."""
        super().__init__()
        self.__trigger = trigger # type: Trigger
        
    def requires_stop(self) -> bool:
        return False

    def build_sequence_loop(self, 
                            delegator: SequencingElement,
                            body: SequencingElement,
                            sequencer: Sequencer,
                            parameters: Dict[str, Parameter],
                            instruction_block: InstructionBlock) -> None:
        body_block = instruction_block.create_embedded_block()
        body_block.return_ip = InstructionPointer(instruction_block, len(body_block))
        
        instruction_block.add_instruction_cjmp(self.__trigger, body_block)
        sequencer.push(body, parameters, body_block)
        
    def build_sequence_branch(self, 
                              delegator: SequencingElement,
                              if_branch: SequencingElement,
                              else_branch: SequencingElement,
                              sequencer: Sequencer,
                              parameters: Dict[str, Parameter],
                              instruction_block: InstructionBlock) -> None:
        if_block = instruction_block.create_embedded_block()
        else_block = instruction_block.create_embedded_block()
        
        instruction_block.add_instruction_cjmp(self.__trigger, if_block)
        sequencer.push(if_branch, parameters, if_block)
        
        instruction_block.add_instruction_goto(else_block)
        sequencer.push(else_branch, parameters, else_block)
        
        if_block.return_ip = InstructionPointer(instruction_block, len(instruction_block))
        else_block.return_ip = if_block.return_ip

    
class SoftwareCondition(Condition):
    """A condition that will be evaluated in the software.
    
    SoftwareConditions are evaluated in software, allowing them to rely on sophisticated measurement evaluation or
    to be used when the hardware device does not support trigger based jumping instructions.
    
    On the downside, this means that a translation processes may be interrupted because a SoftwareCondition
    relying on measurement data cannot be evaluated before that data is acquired. In this case, the already translated
    part has to be executed, the measurement is made and in a subsequent translation, the SoftwareCondition is evaluated
    and the corresponding instructions of one branch/the loop body are generated without jumping instructions.
    
    This interruption of pulse execution might not be feasible in some environments.
    """
        
    def __init__(self, evaluation_callback: Callable[[int], Optional[bool]]) -> None:
        """Create a new SoftwareCondition instance.
        
        Argument evaluationCallback is a callable function which accepts an integer argument and returns a bool or None.
        The integer argument is the current iteration of a loop (starting at zero before the first loop execution). For
        branch sequencing, this argument will always be zero. The callbacks return value must be None, if evaluation
        is currently not possible and boolean otherwise.
        """
        super().__init__()
        self.__callback = evaluation_callback # type: Callable[[int], Optional[bool]]
        self.__loop_iteration = 0
        
    def requires_stop(self) -> bool:
        evaluation_result = self.__callback(self.__loop_iteration)
        return evaluation_result is None

    def build_sequence_loop(self, 
                            delegator: SequencingElement,
                            body: SequencingElement,
                            sequencer: Sequencer,
                            parameters: Dict[str, Parameter],
                            instruction_block: InstructionBlock) -> None:
        
        evaluation_result = self.__callback(self.__loop_iteration)
        if evaluation_result is None:
            raise ConditionEvaluationException()
        #if evaluationResult is None:
        #    instruction_block.add_instruction_stop()
        #    sequencer.push(delegator, time_parameters, voltage_parameters, instruction_block)
        #else:
        # the above should be done by Sequencer via evaluating requires_stop()
        if evaluation_result == True:
            sequencer.push(delegator, parameters, instruction_block)
            sequencer.push(body, parameters, instruction_block)
            self.__loop_iteration += 1 # next time, evaluate for next iteration

    def build_sequence_branch(self, 
                              delegator: SequencingElement,
                              if_branch: SequencingElement,
                              else_branch: SequencingElement,
                              sequencer: Sequencer,
                              parameters: Dict[str, Parameter],
                              instruction_block: InstructionBlock) -> None:
        
        evaluation_result = self.__callback(self.__loop_iteration)
        if evaluation_result is None:
            raise ConditionEvaluationException()
        #if evaluationResult is None:
        #    instruction_block.add_instruction_stop()
        #    sequencer.push(delegator, time_parameters, voltage_parameters, instruction_block)
        #else:
        # the above should be done by Sequencer via evaluating requires_stop()
        if evaluation_result == True:
            sequencer.push(if_branch, parameters, instruction_block)
        else:
            sequencer.push(else_branch, parameters, instruction_block)


class ProxyCondition(Condition):

    def __init__(self, condition_name: str) -> None:
        super().__init__()
        self.__condition_name = condition_name
        self.__condition = None # type: Condition

    def acquire_proxied(self, conditions: Dict[str, Condition]) -> None:
        self.__condition = conditions[self.__condition_name]

    def requires_stop(self) -> bool:
        if self.__condition is None:
            raise Exception("The Condition reference '{}' has not been resolved.".format(self.__condition_name))
        return self.__condition.requires_stop()

    def build_sequence_loop(self,
                            delegator: SequencingElement,
                            body: SequencingElement,
                            sequencer: Sequencer,
                            parameters: Dict[str, Parameter],
                            instruction_block: InstructionBlock) -> None:
        if self.__condition is None:
            raise Exception("The Condition reference '{}' has not been resolved.".format(self.__condition_name))
        self.__condition.build_sequence_loop(delegator, body, sequencer, parameters, instruction_block)

    def build_sequence_branch(self,
                              delegator: SequencingElement,
                              if_branch: SequencingElement,
                              else_branch: SequencingElement,
                              sequencer: Sequencer,
                              parameters: Dict[str, Parameter],
                              instruction_block: InstructionBlock) -> None:
        if self.__condition is None:
            raise Exception("The Condition reference '{}' has not been resolved.".format(self.__condition_name))
        self.__condition.build_sequence_branch(delegator, if_branch, else_branch, sequencer, parameters, instruction_block)


class ConditionEvaluationException(Exception):
    """Indicates that a SoftwareCondition cannot be evaluated yet."""
    
    def __str__(self) -> str:
        return "The Condition can currently not be evaluated."