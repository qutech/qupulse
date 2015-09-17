import logging
from typing import Dict, Set, List

"""RELATED THIRD PARTY IMPORTS"""

"""LOCAL IMPORTS"""
from .Parameter import Parameter
from .PulseTemplate import PulseTemplate, MeasurementWindow

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
    def __init__(self) -> None:
        super().__init__()
        self.else_branch = None
        self.if_branch = None
        raise NotImplementedError()

    def __str__(self) -> str:
        raise NotImplementedError()
    
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


