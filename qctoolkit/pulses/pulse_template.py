"""This module defines the abstract PulseTemplate class which is the basis of any
pulse model in the qctoolkit.

Classes:
    - PulseTemplate: Represents the parametrized general structure of a pulse.
    - AtomicPulseTemplate: PulseTemplate that does imply any control flow disruptions and can be
        directly translated into a waveform.
"""
from abc import ABCMeta, abstractmethod, abstractproperty
from typing import Dict, List, Tuple, Set, Optional, Union
import itertools

MeasurementWindow = Tuple[str, float, float]
ChannelID = Union[str, int]

from qctoolkit.serialization import Serializable

from qctoolkit.pulses.parameters import ParameterDeclaration, Parameter
from qctoolkit.pulses.sequencing import SequencingElement, InstructionBlock

__all__ = ["MeasurementWindow", "PulseTemplate", "AtomicPulseTemplate", "DoubleParameterNameException", "ChannelID"]


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

    @abstractproperty
    def measurement_names(self) -> Set[str]:
        """The set of measurement identifiers in this pulse template"""

    @abstractproperty
    def is_interruptable(self) -> bool:
        """Return true, if this PulseTemplate contains points at which it can halt if interrupted.
        """

    @abstractproperty
    def defined_channels(self) -> Set['ChannelID']:
        """Returns the number of hardware output channels this PulseTemplate defines."""

    def __matmul__(self, other: 'PulseTemplate') -> 'SequencePulseTemplate':
        """This method enables us to use the @-operator (intended for matrix multiplication) for
         concatenating pulses. If one of the pulses is a SequencePulseTemplate the other pulse gets merged into it"""

        from qctoolkit.pulses.sequence_pulse_template import SequencePulseTemplate

        if self != other:
            # check if parameter names of the subpulses intersect and raise an exception if so
            double_parameters = self.parameter_names & other.parameter_names
            if double_parameters:
                raise DoubleParameterNameException(self, other, double_parameters)

        external_parameters = self.parameter_names | other.parameter_names
        subtemplates = itertools.chain(self.subtemplates if isinstance(self, SequencePulseTemplate) else [self],
                                       other.subtemplates if isinstance(other, SequencePulseTemplate) else [other])
        return SequencePulseTemplate(subtemplates, external_parameters)


class PossiblyAtomicPulseTemplate(PulseTemplate):
    """This PulseTemplate may be atomic."""
    def __init__(self, identifier: Optional[str]=None):
        super().__init__(identifier=identifier)

    @abstractmethod
    def build_waveform(self,
                       parameters: Dict[str, Parameter],
                       measurement_mapping: Dict[str, str],
                       channel_mapping: Dict[ChannelID, ChannelID]) -> Optional['Waveform']:
        """Translate this PulseTemplate into a waveform according to the given parameteres.

        Args:
            parameters (Dict(str -> Parameter)): A mapping of parameter names to Parameter objects.
        Returns:
            Waveform object represented by this AtomicPulseTemplate object or None, if this object
                does not represent a valid waveform.
        """

    def atomic_build_sequence(self,
                              parameters: Dict[str, Parameter],
                              measurement_mapping: Dict[str, str],
                              channel_mapping: Dict['ChannelID', 'ChannelID'],
                              instruction_block: InstructionBlock):
        waveform = self.build_waveform(parameters,
                                       measurement_mapping=measurement_mapping,
                                       channel_mapping=channel_mapping)
        if waveform:
            instruction_block.add_instruction_exec(waveform)


class AtomicPulseTemplate(PossiblyAtomicPulseTemplate, metaclass=ABCMeta):
    """A PulseTemplate that does not imply any control flow disruptions and can be directly
    translated into a waveform.

    Implies that no AtomicPulseTemplate object is interruptable.
    """
    def __init__(self, identifier: Optional[str]=None):
        super().__init__(identifier=identifier)

    def is_interruptable(self) -> bool:
        return False

    @property
    def atomicity(self) -> bool:
        return True

    def build_sequence(self,
                       sequencer: 'Sequencer',
                       parameters: Dict[str, Parameter],
                       conditions: Dict[str, 'Condition'],
                       measurement_mapping: Dict[str, str],
                       channel_mapping: Dict['ChannelID', 'ChannelID'],
                       instruction_block: InstructionBlock) -> None:
        self.atomic_build_sequence(parameters=parameters,
                                   measurement_mapping=measurement_mapping,
                                   channel_mapping=channel_mapping,
                                   instruction_block=instruction_block)


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

