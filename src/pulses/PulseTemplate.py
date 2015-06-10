"""STANDARD LIBRARY IMPORTS"""
from abc import ABCMeta, abstractmethod
from typing import Dict

"""RELATED THIRD PARTY IMPORTS"""

"""LOCAL IMPORTS"""
from Parameter import ParameterDeclaration


class PulseTemplate(metaclass = ABCMeta):
    """docstring for PulseTemplate"""
    def __init__(self):
        super().__init__()
        
    @abstractmethod
    def __len__(self) -> int:
        """Defines the behaviour of len(PulseTemplate), which is the sum of all subpulses. 
        __len__ already provides a type check to assure that only numerical values are returned
        """
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, ParameterDeclaration]:
        pass

    @abstractmethod
    def get_measurement_windows(self):
        pass

    @abstractmethod
    def is_interruptable(self) -> bool:
        pass

    @abstractmethod
    def generate_waveforms(self, sequencer, parameters):
        """Compile a waveform of the pulse represented by this PulseTemplate and the given parameters using the hardware-specific Sequencer object."""
        pass

    @abstractmethod
    def set_parameter(self, name, value):
        pass


class ParameterNotInPulseTemplateException(Exception):
    """docstring for ParameterNotInPulseException"""
    def __init__(self, name: str, pulse_template: PulseTemplate):
        super().__init__()
        self.name = name
        self.pulse_template = pulse_template

    def __str__(self):
        return "Parameter {1} not found".format(self.name)