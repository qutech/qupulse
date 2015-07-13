"""STANDARD LIBRARY IMPORTS"""
import logging
from typing import Dict, Set, List

"""RELATED THIRD PARTY IMPORTS"""

"""LOCAL IMPORTS"""
from .Parameter import ParameterDeclaration, TimeParameterDeclaration, Parameter
from .PulseTemplate import PulseTemplate, MeasurementWindow

logger = logging.getLogger(__name__)

class BranchPulseTemplate(PulseTemplate):
    """!@brief Conditional branching in a pulse.
    
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
    
    def get_time_parameter_names(self) -> Set[str]:
        """!@brief Return the set of names of declared time parameters."""
        raise NotImplementedError()
        
    def get_voltage_parameter_names(self) -> Set[str]:
        """!@brief Return the set of names of declared voltage parameters."""
        raise NotImplementedError()
        
    def get_time_parameter_declarations(self) -> Dict[str, TimeParameterDeclaration]:
        """!@brief Return a copy of the dictionary containing the time parameter declarations of this PulseTemplate."""
        raise NotImplementedError()
        
    def get_voltage_parameter_declarations(self) -> Dict[str, ParameterDeclaration]:
        """!@brief Return a copy of the dictionary containing the voltage parameter declarations of this PulseTemplate."""
        raise NotImplementedError()

    def get_measurement_windows(self, time_parameters: Dict[str, Parameter] = None) -> List[MeasurementWindow]:
        """!@brief Return all measurement windows defined in this PulseTemplate."""
        raise NotImplementedError()

    def is_interruptable(self) -> bool:
        """!@brief Return true, if this PulseTemplate contains points at which it can halt if interrupted."""
        raise NotImplementedError()


