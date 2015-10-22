
from abc import ABCMeta, abstractmethod, abstractproperty
from typing import Dict, List, Tuple, Set, Optional
import logging

"""RELATED THIRD PARTY IMPORTS"""

"""LOCAL IMPORTS"""
from qctoolkit.serialization import Serializable

from .parameters import ParameterDeclaration, Parameter
from .sequencing import SequencingElement

logger = logging.getLogger(__name__)

MeasurementWindow = Tuple[float, float]


class PulseTemplate(Serializable, SequencingElement, metaclass = ABCMeta):
    """A PulseTemplate represents the parameterized general structure of a pulse.

    A PulseTemplate described a pulse in an abstract way: It defines the structure of a pulse
    but might leave some timings or voltage levels undefined, thus declaring parameters.
    This allows to reuse a PulseTemplate for several pulses which have the same overall structure
    and differ only in concrete values for the parameters.
    Obtaining an actual pulse which can be executed by specifying values for these parameters is
    called instantiation of the PulseTemplate.
    """

    def __init__(self, identifier: Optional[str]=None) -> None:
        super().__init__(identifier)

    @abstractproperty
    def parameter_names(self) -> Set[str]:
        """Return the set of names of declared parameters."""

    @abstractproperty
    def parameter_declarations(self) -> Set[ParameterDeclaration]:
        """Return the set of ParameterDeclarations."""

    @abstractmethod
    def get_measurement_windows(self, parameters: Dict[str, Parameter] = None) -> List[MeasurementWindow]:
        """Return all measurement windows defined in this PulseTemplate."""

    @abstractproperty
    def is_interruptable(self) -> bool:
        """Return true, if this PulseTemplate contains points at which it can halt if interrupted."""


class ParameterNotInPulseTemplateException(Exception):
    """Indicates that a provided parameter was not declared in a PulseTemplate."""

    def __init__(self, name: str, pulse_template: PulseTemplate) -> None:
        super().__init__()
        self.name = name
        self.pulse_template = pulse_template

    def __str__(self) -> str:
        return "Parameter {1} not found in pulse template {2}".format(self.name, self.pulse_template)