import logging
from typing import Dict, Set, List

"""RELATED THIRD PARTY IMPORTS"""

"""LOCAL IMPORTS"""
from .parameters import Parameter
from .pulse_template import PulseTemplate, MeasurementWindow

logger = logging.getLogger(__name__)

class LoopPulseTemplate(PulseTemplate):
    """Conditional looping in a pulse.
    
    A LoopPulseTemplate is a PulseTemplate which is repeated
    during execution as long as a certain condition holds.
    """
    
    def __init__(self) -> None:
        super().__init__()
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
        """Return true, if this PulseTemplate contains points at which it can halt if interrupted."""
        raise NotImplementedError()
        