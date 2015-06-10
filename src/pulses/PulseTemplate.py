"""STANDARD LIBRARY IMPORTS"""
from logging import getLogger, Logger
from abc import ABCMeta, abstractmethod

"""RELATED THIRD PARTY IMPORTS"""

"""LOCAL IMPORTS"""


class PulseTemplate(metaclass = ABCMeta):
    """docstring for PulseTemplate"""
    def __init__(self):
        super(PulseTemplate, self).__init__()
        

    @abstractmethod
    def __len__(self) -> int:
        """Defines the behaviour of len(PulseTemplate), which is the sum of all subpulses. 
        __len__ already provides a type check to assure that only numerical values are returned
        """
        pass

    @abstractmethod
    def __str__(self) -> string:
        pass
    
    @abstractmethod
    def get_parameters(self) -> {string,Parameter}:
        pass

    @abstractmethod
    def get_measurement_windows(self):
        pass

    @abstractmethod
    def is_interruptable(self) -> bool:
        pass

    @abstractmethod
    def generate_waveforms(self, sequencer) -> Waveform:
        pass

    @abstractmethod
    def set_parameter(self, name, value):
        pass


class ParameterNotInPulseTemplateException(Exception):
    """docstring for ParameterNotInPulseException"""
    def __init__(self, name: string, pulse_template: PulseTemplate):
        super(ParameterNotInPulseException, self).__init__()
        self.name = name
        self.pulse_template = pulse_template

    def __str__(self):
        return "Parameter {1} not found".format(name)