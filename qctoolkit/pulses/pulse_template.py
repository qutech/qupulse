"""This module defines the abstract PulseTemplate class which is the basis of any
pulse model in the qctoolkit.

Classes:
    - PulseTemplate: Represents the parametrized general structure of a pulse
"""
from abc import ABCMeta, abstractmethod, abstractproperty
from typing import Dict, List, Tuple, Set, Optional

from qctoolkit.serialization import Serializable

from qctoolkit.pulses.parameters import ParameterDeclaration, Parameter
from qctoolkit.pulses.sequencing import SequencingElement

__all__ = ["MeasurementWindow", "PulseTemplate", "DoubleParameterNameException"]


MeasurementWindow = Tuple[float, float]


class PulseTemplate(Serializable, SequencingElement, metaclass=ABCMeta):
    """A PulseTemplate represents the parametrized general structure of a pulse.

    A PulseTemplate described a pulse in an abstract way: It defines the structure of a pulse
    but might leave some timings or voltage levels undefined, thus declaring parameters.
    This allows to reuse a PulseTemplate for several pulses which have the same overall structure
    and differ only in concrete values for the parameters.
    Obtaining an actual pulse which can be executed by specifying values for these parameters is
    called instantiation of the PulseTemplate and achieved by invoking the sequencing process.
    """

    def __init__(self, identifier: Optional[str]=None) -> None:
        super().__init__(identifier)

    @abstractproperty
    def parameter_names(self) -> Set[str]:
        """The set of names of parameters required to instantiate this PulseTemplate."""

    @abstractproperty
    def parameter_declarations(self) -> Set[ParameterDeclaration]:
        """The set of ParameterDeclaration objects detailing all parameters required to instantiate
        this PulseTemplate.
        """

    @abstractmethod
    def get_measurement_windows(self, parameters: Dict[str, Parameter]=None) \
            -> List[MeasurementWindow]:
        """
        FLAWED / OBSOLETE: should be fixed already in a different branch and will be merged soon

        Returns:
             All measurement windows defined in this PulseTemplate.
         """

    @abstractproperty
    def is_interruptable(self) -> bool:
        """Return true, if this PulseTemplate contains points at which it can halt if interrupted.
        """

    def __matmult__(self, other) -> 'SequencePulseTemplate':
        """This method enables us to use the @-operator (intended for matrix multiplication) for
         concatenating pulses"""
        from qctoolkit.pulses.sequence_pulse_template import SequencePulseTemplate
        # check if parameter names of the subpulses clash, otherwise construct a default mapping
        double_parameters = self.parameter_names & other.parameter_names # intersection
        if double_parameters:
            # if there are parameter name conflicts, throw an exception
            raise DoubleParameterNameException(self, other, double_parameters)
        else:
            subtemplates = [(self, {p:p for p in self.parameter_names}),
                            (other, {p:p for p in other.parameter_names})]
            external_parameters = self.parameter_names | other.parameter_names # union
            return SequencePulseTemplate(subtemplates, external_parameters)


class DoubleParameterNameException(Exception):

    def __init__(self, templateA: PulseTemplate, templateB: PulseTemplate, names: Set[str]) -> None:
        super().__init__()
        self.templateA = templateA
        self.templateB = templateB
        self.names = names

    def __str__(self) -> str:
        return "Cannot concatenate pulses '{}' and '{}' with a default parameter mapping. " \
               "Both define the following parameter names: {}".format(
            self.templateA, self.templateB, ', '.join(self.names)
        )


