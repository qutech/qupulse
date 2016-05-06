import logging
from typing import Dict, Set, List, Optional, Any

"""RELATED THIRD PARTY IMPORTS"""

"""LOCAL IMPORTS"""
from qctoolkit.pulses.parameters import Parameter
from qctoolkit.pulses.pulse_template import PulseTemplate, MeasurementWindow
from qctoolkit.pulses.conditions import Condition, ConditionMissingException
from qctoolkit.pulses.sequencing import Sequencer, InstructionBlock

__all__ = ["BranchPulseTemplate"]

logger = logging.getLogger(__name__)


class BranchPulseTemplate(PulseTemplate):
    """Conditional branching in a pulse.
    
    A BranchPulseTemplate is a PulseTemplate
    with different structures depending on a certain condition.
    It defines refers to an if-branch and an else-branch, which
    are both PulseTemplates.
    When instantiating a pulse from a BranchPulseTemplate,
    both branches refer to concrete pulses. If the given
    condition evaluates to true at the time the pulse is executed,
    the if-branch, otherwise the else-branch, is chosen for execution.
    This allows for alternative execution 'paths' in pulses.
    
    Both branches must be of the same length.
    """
    def __init__(self, condition: str, if_branch: PulseTemplate, else_branch: PulseTemplate, identifier: Optional[str]=None) -> None:
        super().__init__(identifier=identifier)
        self.__condition = condition
        self.__if_branch = if_branch
        self.__else_branch = else_branch

    def __str__(self) -> str:
        return "BranchPulseTemplate: Condition <{}>, If-Branch <{}>, Else-Branch <{}>".format(self.__condition, self.__if_branch, self.__else_branch)
    
    @property
    def parameter_names(self) -> Set[str]:
        """Return the set of names of declared parameters."""
        raise NotImplementedError()

    @property
    def parameter_declarations(self) -> Set[str]:
        """Return the set of ParameterDeclarations."""
        raise NotImplementedError()

    def get_measurement_windows(self, parameters: Dict[str, Parameter] = None) -> List[MeasurementWindow]:
        """Return all measurement windows defined in this PulseTemplate."""
        raise NotImplementedError()

    @property
    def is_interruptable(self) -> bool:
        """Return True, if this PulseTemplate contains points at which it can halt if interrupted."""
        raise NotImplementedError()

    def __obtain_condition_object(self, conditions: Dict[str, Condition]) -> Condition:
        try:
            return conditions[self.__condition]
        except:
            ConditionMissingException(self.__condition)

    def build_sequence(self,
                       sequencer: Sequencer,
                       parameters: Dict[str, Parameter],
                       conditions: Dict[str, Condition],
                       instruction_block: InstructionBlock) -> None:
        self.__obtain_condition_object(conditions).build_sequence_branch(self,
                                                                         self.__if_branch,
                                                                         self.__else_branch,
                                                                         sequencer,
                                                                         parameters,
                                                                         conditions,
                                                                         instruction_block)

    def requires_stop(self, parameters: Dict[str, Parameter], conditions: Dict[str, Condition]) -> bool:
        return self.__obtain_condition_object(conditions).requires_stop()

    def get_serialization_data(self, serializer: 'Serializer') -> Dict[str, Any]:
        raise NotImplementedError()

    @staticmethod
    def deserialize(serializer: 'Serializer', **kwargs) -> 'BranchPulseTemplate':
        raise NotImplementedError()

