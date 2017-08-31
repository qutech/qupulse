"""This module defines conditions required for branching decisions in pulse execution.

Classes:
    - Condition: Base-class for conditions.
    - SoftwareCondition: A software-evaluated condition.
    - HardwareCondition: A hardware-evaluated condition.
    - ConditionEvaluationException.
    - ConditionMissingException.
"""

from abc import ABCMeta, abstractmethod
from typing import Dict, Optional, Callable

from qctoolkit.utils.types import ChannelID
from qctoolkit.pulses.parameters import Parameter
from . import sequencing
from qctoolkit.pulses.instructions import InstructionBlock, InstructionPointer, Trigger

__all__ = ["Condition", "ConditionEvaluationException", "ConditionMissingException",
           "SoftwareCondition", "HardwareCondition"]


class Condition(metaclass=ABCMeta):
    """A condition on which the execution of a pulse may depend.
    
    Conditions are used for branching and looping of pulses and
    thus relevant for BranchPulseTemplate and LoopPulseTemplate.
    Implementations of Condition may rely on software variables,
    measured data or be mere placeholders for hardware triggers.
    """
    def __init__(self) -> None:
        super().__init__()
        
    @abstractmethod
    def requires_stop(self) -> bool:
        """Query whether evaluating this Condition object requires an interruption in execution/
        sequencing,  e.g. because it depends on a value obtained during executin.

        Returns:
            True, if evaluation of this Condition object requires sequencing to be interrupted.
        """
        
    @abstractmethod
    def build_sequence_loop(self, 
                            delegator: sequencing.SequencingElement,
                            body: sequencing.SequencingElement,
                            sequencer: sequencing.Sequencer,
                            parameters: Dict[str, Parameter],
                            conditions: Dict[str, 'Condition'],
                            measurement_mapping: Dict[str, str],
                            channel_mapping: Dict[ChannelID, ChannelID],
                            instruction_block: InstructionBlock) -> None:
        """Translate a looping SequencingElement using this Condition into an instruction sequence
        for the given instruction block using sequencer and the given parameter sets.

        Args:
            delegator: The SequencingElement which has delegated the invocation
                of its build_sequence method to this Condition object.
            body: The SequencingElement representing the loops body.
            sequencer: The Sequencer object coordinating the current sequencing process.
            parameters: A mapping of parameter names to Parameter objects
                which will be passed to the loop body.
            conditions: A mapping of condition identifier to Condition
                objects which will be passed to the loop body.
            instruction_block: The instruction block into which instructions
                resulting from the translation of this Condition object will be placed.
        See Also:
            SequencingElement.build_sequence()
        """
    
    @abstractmethod
    def build_sequence_branch(self,
                              delegator: sequencing.SequencingElement,
                              if_branch: sequencing.SequencingElement,
                              else_branch: sequencing.SequencingElement,
                              sequencer: sequencing.Sequencer,
                              parameters: Dict[str, Parameter],
                              conditions: Dict[str, 'Condition'],
                              measurement_mapping: Dict[str, str],
                              channel_mapping: Dict['ChannelID', 'ChannelID'],
                              instruction_block: InstructionBlock) -> None:
        """Translate a branching SequencingElement using this Condition into an instruction sequence
        for the given instruction block using sequencer and the given parameter sets.

        Args:
           delegator: The SequencingElement which has delegated the invocation
               of its build_sequence method to this Condition object.
           if_branch: The SequencingElement representing the branch executed
               if the condition holds.
           else_branch: The SequencingElement representing the branch executed
               if the condition does not hold.
           parameters: A mapping of parameter names to Parameter objects
               which will be passed to the loop body.
           conditions: A mapping of condition identifier to Condition
               objects which will be passed to the loop body.
           instruction_block: The instruction block into which instructions
               resulting from the translation of this Condition object will be placed.

        See Also:
            SequencingElement.build_sequence()
        """


class HardwareCondition(Condition):
    """A condition that will be evaluated using hardware triggers.

    The condition will be evaluate as true iff the trigger has fired before the hardware device
    makes the branching decision.

    During the translation process, a HardwareCondition instance will produce code blocks for
    branches/loop bodies and the corresponding conditional jump instructions.
    """
    
    def __init__(self, trigger: Trigger) -> None:
        """Create a new HardwareCondition instance.

        Args:
             trigger: The trigger handle of the corresponding hardware device."""
        super().__init__()
        self.__trigger = trigger  # type: Trigger
        
    def requires_stop(self) -> bool:
        return False

    def build_sequence_loop(self, 
                            delegator: sequencing.SequencingElement,
                            body: sequencing.SequencingElement,
                            sequencer: sequencing.Sequencer,
                            parameters: Dict[str, Parameter],
                            conditions: Dict[str, Condition],
                            measurement_mapping: Dict[str, str],
                            channel_mapping: Dict[ChannelID, ChannelID],
                            instruction_block: InstructionBlock) -> None:
        body_block = InstructionBlock()
        body_block.return_ip = InstructionPointer(instruction_block,
                                                  len(instruction_block.instructions))
        
        instruction_block.add_instruction_cjmp(self.__trigger, body_block)
        sequencer.push(body, parameters, conditions, window_mapping=measurement_mapping,
                       channel_mapping=channel_mapping,
                       target_block=body_block)
        
    def build_sequence_branch(self,
                              delegator: sequencing.SequencingElement,
                              if_branch: sequencing.SequencingElement,
                              else_branch: sequencing.SequencingElement,
                              sequencer: sequencing.Sequencer,
                              parameters: Dict[str, Parameter],
                              conditions: Dict[str, Condition],
                              measurement_mapping: Dict[str, str],
                              channel_mapping: Dict[ChannelID, ChannelID],
                              instruction_block: InstructionBlock) -> None:
        if_block = InstructionBlock()
        else_block = InstructionBlock()
        
        instruction_block.add_instruction_cjmp(self.__trigger, if_block)
        sequencer.push(if_branch, parameters, conditions, window_mapping=measurement_mapping,
                       channel_mapping=channel_mapping,
                       target_block=if_block)
        
        instruction_block.add_instruction_goto(else_block)
        sequencer.push(else_branch, parameters, conditions, window_mapping=measurement_mapping,
                       channel_mapping=channel_mapping,
                       target_block=else_block)
        
        if_block.return_ip = InstructionPointer(instruction_block,
                                                len(instruction_block.instructions))
        else_block.return_ip = if_block.return_ip

    
class SoftwareCondition(Condition):
    """A condition that will be evaluated in the software.
    
    SoftwareConditions are evaluated in software, allowing them to rely on sophisticated measurement
    evaluation or to be used when the hardware device does not support trigger based jumping
    instructions.
    
    On the downside, this means that a translation processes may be interrupted because a
    SoftwareCondition relying on measurement data cannot be evaluated before that data is acquired.
    In this case, the already translated part has to be executed, the measurement is made and in a
    subsequent translation, the SoftwareCondition is evaluated and the corresponding instructions
    of one branch/the loop body are generated without jumping instructions.
    
    This interruption of pulse execution might not be feasible in some environments.
    """
        
    def __init__(self, evaluation_callback: Callable[[int], Optional[bool]]) -> None:
        """Create a new SoftwareCondition instance.

        Args:
            evaluation_callback: A function handle which accepts an integer arguments and returns
                a boolean value or None. The integer argument is the current iteration of loop
                (starting at zero before the first loop execution). For branch sequencing, this
                argument will always be zero. The callback's return value must be None iff
                evaluation is currently not possible.
        """
        super().__init__()
        self.__callback = evaluation_callback # type: Callable[[int], Optional[bool]]
        self.__loop_iteration = 0
        
    def requires_stop(self) -> bool:
        evaluation_result = self.__callback(self.__loop_iteration)
        return evaluation_result is None

    def build_sequence_loop(self, 
                            delegator: sequencing.SequencingElement,
                            body: sequencing.SequencingElement,
                            sequencer: sequencing.Sequencer,
                            parameters: Dict[str, Parameter],
                            conditions: Dict[str, Condition],
                            measurement_mapping: Dict[str, str],
                            channel_mapping: Dict[ChannelID, ChannelID],
                            instruction_block: InstructionBlock) -> None:
        
        evaluation_result = self.__callback(self.__loop_iteration)
        if evaluation_result is None:
            raise ConditionEvaluationException()
        if evaluation_result is True:
            sequencer.push(delegator, parameters, conditions, window_mapping=measurement_mapping,
                           channel_mapping=channel_mapping,
                           target_block=instruction_block)
            sequencer.push(body, parameters, conditions, window_mapping=measurement_mapping,
                           channel_mapping=channel_mapping,
                           target_block=instruction_block)
            self.__loop_iteration += 1 # next time, evaluate for next iteration

    def build_sequence_branch(self,
                              delegator: sequencing.SequencingElement,
                              if_branch: sequencing.SequencingElement,
                              else_branch: sequencing.SequencingElement,
                              sequencer: sequencing.Sequencer,
                              parameters: Dict[str, Parameter],
                              conditions: Dict[str, Condition],
                              measurement_mapping: Dict[str, str],
                              channel_mapping: Dict[ChannelID, ChannelID],
                              instruction_block: InstructionBlock) -> None:
        
        evaluation_result = self.__callback(self.__loop_iteration)
        if evaluation_result is None:
            raise ConditionEvaluationException()
        if evaluation_result is True:
            sequencer.push(if_branch, parameters, conditions, window_mapping=measurement_mapping,
                           channel_mapping=channel_mapping,
                           target_block=instruction_block)
        else:
            sequencer.push(else_branch, parameters, conditions, window_mapping=measurement_mapping,
                           channel_mapping=channel_mapping,
                           target_block=instruction_block)


class ConditionEvaluationException(Exception):
    """Indicates that a SoftwareCondition cannot be evaluated yet."""
    
    def __str__(self) -> str:
        return "The Condition can currently not be evaluated."


class ConditionMissingException(Exception):
    """Indicates that a Condition object was not provided for a condition identifier."""

    def __init__(self, condition_name: str) -> None:
        super().__init__()
        self.condition_name = condition_name

    def __str__(self) -> str:
        return "Condition <{}> was referred to but not provided in the conditions dictionary."\
            .format(self.condition_name)
