"""This module defines BranchPulseTemplate, a higher-order hierarchical pulse template that
conditionally executes one out of two possible PulseTemplates."""

from typing import Dict, Set, List, Optional, Any

from qctoolkit.pulses.parameters import Parameter
from qctoolkit.pulses.pulse_template import PulseTemplate, MeasurementWindow
from qctoolkit.pulses.conditions import Condition, ConditionMissingException
from qctoolkit.pulses.sequencing import Sequencer, InstructionBlock
from qctoolkit.serialization import Serializer

__all__ = ["BranchPulseTemplate"]


class BranchPulseTemplate(PulseTemplate):
    """Conditional branching in a pulse.
    
    A BranchPulseTemplate is a PulseTemplate with different structures depending on a certain
    condition. It refers to an if-branch and an else-branch, which are both PulseTemplates.
    When instantiating a pulse from a BranchPulseTemplate, both branches refer to concrete pulses.
    If the given condition evaluates to true at the time the pulse is executed,
    the if-branch, otherwise the else-branch, is chosen for execution.
    This allows for alternative execution 'paths' in pulses.
    
    Both branches must be of the same length.
    """

    def __init__(self,
                 condition: str,
                 if_branch: PulseTemplate,
                 else_branch: PulseTemplate,
                 identifier: Optional[str]=None) -> None:
        """Create a new BranchPulseTemplate instance.

        Args:
            condition (str): A unique identifier for the branching condition. Will be used to obtain
                the Condition object from the mapping passed in during the sequencing process.
            if_branch (PulseTemplate): A PulseTemplate representing the pulse that is to be
                executed if the condition holds.
            else_branch (PulseTemplate): A PulseTemplate representing the pulse that is to be
                executed if the condition does not hold.
            identifier (str): A unique identifier for use in serialization. (optional)
        """
        super().__init__(identifier=identifier)
        if if_branch.num_channels != else_branch.num_channels:
            raise ValueError("The number of channels in the provided pulses differ!")
        self.__condition = condition
        self.__if_branch = if_branch
        self.__else_branch = else_branch

    def __str__(self) -> str:
        return "BranchPulseTemplate: Condition <{}>, If-Branch <{}>, Else-Branch <{}>"\
            .format(self.__condition, self.__if_branch, self.__else_branch)
    
    @property
    def parameter_names(self) -> Set[str]:
        return self.__if_branch.parameter_names | self.__else_branch.parameter_names

    @property
    def parameter_declarations(self) -> Set[str]:
        return self.__if_branch.parameter_declarations | self.__else_branch.parameter_declarations

    @property
    def is_interruptable(self) -> bool:
        return self.__if_branch.is_interruptable and self.__else_branch.is_interruptable

    @property
    def num_channels(self) -> int:
        return self.__if_branch.num_channels

    @property
    def measurement_names(self) -> Set[str]:
        return self.__if_branch.measurement_names | self.__else_branch.measurement_names

    def __obtain_condition_object(self, conditions: Dict[str, Condition]) -> Condition:
        try:
            return conditions[self.__condition]
        except KeyError as key_error:
            raise ConditionMissingException(self.__condition) from key_error

    def build_sequence(self,
                       sequencer: Sequencer,
                       parameters: Dict[str, Parameter],
                       conditions: Dict[str, Condition],
                       measurement_mapping: Dict[str, str],
                       instruction_block: InstructionBlock) -> None:
        self.__obtain_condition_object(conditions).build_sequence_branch(self,
                                                                         self.__if_branch,
                                                                         self.__else_branch,
                                                                         sequencer,
                                                                         parameters,
                                                                         conditions,
                                                                         measurement_mapping,
                                                                         instruction_block)

    def requires_stop(self,
                      parameters: Dict[str, Parameter],
                      conditions: Dict[str, Condition]) -> bool:
        return self.__obtain_condition_object(conditions).requires_stop()

    def get_serialization_data(self, serializer: Serializer) -> Dict[str, Any]:
        return dict(
            if_branch_template=serializer.dictify(self.__if_branch),
            else_branch_template=serializer.dictify(self.__else_branch),
            condition=self.__condition
        )

    @staticmethod
    def deserialize(serializer: Serializer,
                    condition: str,
                    if_branch_template: Dict[str, Any],
                    else_branch_template: Dict[str, Any],
                    identifier: Optional[str]=None) -> 'BranchPulseTemplate':
        return BranchPulseTemplate(condition,
                                   serializer.deserialize(if_branch_template),
                                   serializer.deserialize(else_branch_template),
                                   identifier)
