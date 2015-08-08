"""STANDARD LIBRARY IMPORTS"""
from abc import ABCMeta, abstractmethod
from typing import Dict, List, Tuple, Set
import logging

"""RELATED THIRD PARTY IMPORTS"""

"""LOCAL IMPORTS"""
from .Parameter import ParameterDeclaration, TimeParameterDeclaration, Parameter
from .Sequencer import SequencingElement

logger = logging.getLogger(__name__)

MeasurementWindow = Tuple[int, int]

class PulseTemplate(SequencingElement, metaclass = ABCMeta):
    """A PulseTemplate represents the parametrized general structure of a pulse.
    
    A PulseTemplate described a pulse in an abstract way: It defines the structure of a pulse
    but might leave some timings or voltage levels undefined, thus declaring parameters.
    This allows to reuse a PulseTemplate for several pulses which have the same overall structure
    and differ only in concrete values for the parameters.
    Obtaining an actual pulse which can be executed by specifying values for these parameters is
    called instantiation of the PulseTemplate.
    """
    
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def __str__(self) -> str:
        pass
    
    @abstractmethod
    def get_time_parameter_names(self) -> Set[str]:
        """Return the set of names of declared time parameters."""
        pass
        
    @abstractmethod
    def get_voltage_parameter_names(self) -> Set[str]:
        """Return the set of names of declared voltage parameters."""
        pass
        
    @abstractmethod
    def get_time_parameter_declarations(self) -> Dict[str, TimeParameterDeclaration]:
        """Return a copy of the dictionary containing the time parameter declarations of this PulseTemplate."""
        pass
        
    @abstractmethod
    def get_voltage_parameter_declarations(self) -> Dict[str, ParameterDeclaration]:
        """Return a copy of the dictionary containing the voltage parameter declarations of this PulseTemplate."""
        pass

    @abstractmethod
    def get_measurement_windows(self, time_parameters: Dict[str, Parameter] = None) -> List[MeasurementWindow]:
        """Return all measurement windows defined in this PulseTemplate."""
        pass

    @abstractmethod
    def is_interruptable(self) -> bool:
        """Return true, if this PulseTemplate contains points at which it can halt if interrupted."""
        pass


class ParameterNotInPulseTemplateException(Exception):
    """Indicates that a provided parameter was not declared in a PulseTemplate."""
    
    def __init__(self, name: str, pulse_template: PulseTemplate) -> None:
        super().__init__()
        self.name = name
        self.pulse_template = pulse_template

    def __str__(self) -> str:
        return "Parameter {1} not found".format(self.name)