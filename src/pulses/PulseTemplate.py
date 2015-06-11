"""STANDARD LIBRARY IMPORTS"""
from abc import ABCMeta, abstractmethod
from typing import Dict, List, Tuple, Set
import logging

"""RELATED THIRD PARTY IMPORTS"""

"""LOCAL IMPORTS"""
from Parameter import ParameterDeclaration, Parameter

logger = logging.getLogger(__name__)

class PulseTemplate(metaclass = ABCMeta):
    """docstring for PulseTemplate"""
    def __init__(self):
        super().__init__()
        
    @abstractmethod
    def __len__(self) -> int:
        """Defines the behaviour of len(PulseTemplate), which is the sum of all subpulses. 
        __len__ already provides a type check to assure that only numerical values are returned
        """
        # TODO: decide whether or not measuring the length of a PulseTemplate actually makes sense, since it may depend on parameters (maybe move to Pulse)
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass
    
    @abstractmethod
    def get_parameter_names(self) -> Set[str]:
        """Return the set of names of declared parameters."""
        pass
        
    @abstractmethod
    def get_parameter_declarations(self) -> Dict[str, ParameterDeclaration]:
        """Return a copy of the dictionary containing the parameter declarations of this PulseTemplate."""
        pass

    @abstractmethod
    def get_measurement_windows(self) -> List[Tuple[float, float]]:
        """Return all measurment windows defined in this PulseTemplate."""
        # TODO: decide whether or not defining exact measurment windows on templates makes sense (maybe move to Pulse)
        pass

    @abstractmethod
    def is_interruptable(self) -> bool:
        """Return true, if this PulseTemplate contains points at which it can halt if interrupted."""
        pass

    @abstractmethod
    def generate_waveforms(self, sequencer: "Sequencer", parameters: Dict[str, Parameter]) -> None:
        """Compile a waveform of the pulse represented by this PulseTemplate and the given parameters using the hardware-specific Sequencer object."""
        pass


class ParameterNotInPulseTemplateException(Exception):
    """docstring for ParameterNotInPulseException"""
    def __init__(self, name: str, pulse_template: PulseTemplate):
        super().__init__()
        self.name = name
        self.pulse_template = pulse_template

    def __str__(self):
        return "Parameter {1} not found".format(self.name)