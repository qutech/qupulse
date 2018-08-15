"""This module defines the abstract PulseTemplate class which is the basis of any
pulse model in the qctoolkit.

Classes:
    - PulseTemplate: Represents the parametrized general structure of a pulse.
    - AtomicPulseTemplate: PulseTemplate that does imply any control flow disruptions and can be
        directly translated into a waveform.
"""
from abc import abstractmethod
from typing import Dict, Tuple, Set, Optional, Union, List, Callable, Any, Generic, TypeVar, Mapping
import itertools
from contextlib import contextmanager
from numbers import Real

from qctoolkit.utils.types import ChannelID, DocStringABCMeta
from qctoolkit.serialization import Serializable
from qctoolkit.expressions import ExpressionScalar
from qctoolkit._program._loop import Loop, to_waveform
from qctoolkit._program.transformation import Transformation, IdentityTransformation, ChainedTransformation, chain_transformations


from qctoolkit.pulses.conditions import Condition
from qctoolkit.pulses.parameters import Parameter, ConstantParameter, ParameterNotProvidedException
from qctoolkit.pulses.sequencing import Sequencer, SequencingElement, InstructionBlock
from qctoolkit._program.waveforms import Waveform, TransformingWaveform
from qctoolkit.pulses.measurement import MeasurementDefiner, MeasurementDeclaration

__all__ = ["PulseTemplate", "AtomicPulseTemplate", "DoubleParameterNameException", "MappingTuple"]


MappingTuple = Union[Tuple['PulseTemplate'],
                     Tuple['PulseTemplate', Dict],
                     Tuple['PulseTemplate', Dict, Dict],
                     Tuple['PulseTemplate', Dict, Dict, Dict]]


class PulseTemplate(Serializable, SequencingElement, metaclass=DocStringABCMeta):
    """A PulseTemplate represents the parametrized general structure of a pulse.

    A PulseTemplate described a pulse in an abstract way: It defines the structure of a pulse
    but might leave some timings or voltage levels undefined, thus declaring parameters.
    This allows to reuse a PulseTemplate for several pulses which have the same overall structure
    and differ only in concrete values for the parameters.
    Obtaining an actual pulse which can be executed by specifying values for these parameters is
    called instantiation of the PulseTemplate and achieved by invoking the sequencing process.
    """

    def __init__(self, *,
                 identifier: Optional[str]) -> None:
        super().__init__(identifier=identifier)

    @property
    @abstractmethod
    def parameter_names(self) -> Set[str]:
        """The set of names of parameters required to instantiate this PulseTemplate."""

    @property
    @abstractmethod
    def measurement_names(self) -> Set[str]:
        """The set of measurement identifiers in this pulse template."""

    @property
    @abstractmethod
    def is_interruptable(self) -> bool:
        """Return true, if this PulseTemplate contains points at which it can halt if interrupted.
        """

    @property
    @abstractmethod
    def duration(self) -> ExpressionScalar:
        """An expression for the duration of this PulseTemplate."""

    @property
    @abstractmethod
    def defined_channels(self) -> Set['ChannelID']:
        """Returns the number of hardware output channels this PulseTemplate defines."""

    @property
    def num_channels(self) -> int:
        """The number of channels this PulseTemplate defines"""
        return len(self.defined_channels)

    def __matmul__(self, other: Union['PulseTemplate', MappingTuple]) -> 'SequencePulseTemplate':
        """This method enables using the @-operator (intended for matrix multiplication) for
         concatenating pulses. If one of the pulses is a SequencePulseTemplate the other pulse gets merged into it"""
        from qctoolkit.pulses.sequence_pulse_template import SequencePulseTemplate

        return SequencePulseTemplate.concatenate(self, other)

    def __rmatmul__(self, other: MappingTuple) -> 'SequencePulseTemplate':
        from qctoolkit.pulses.sequence_pulse_template import SequencePulseTemplate

        return SequencePulseTemplate.concatenate(other, self)

    @property
    @abstractmethod
    def integral(self) -> Dict[ChannelID, ExpressionScalar]:
        """Returns an expression giving the integral over the pulse."""

    def create_program(self, *,
                       parameters: Optional[Dict[str, Parameter]]=None,
                       measurement_mapping: Optional[Dict[str, Optional[str]]]=None,
                       channel_mapping: Optional[Dict[ChannelID, Optional[ChannelID]]]=None,
                       global_transformation: Optional[Transformation]=None,
                       to_single_waveform: Set[Union[str, 'PulseTemplate']]=None) -> Optional['Loop']:
        """Translates this PulseTemplate into a program Loop.

        The returned Loop represents the PulseTemplate with all parameter values instantiated provided as dictated by
        the parameters argument. Optionally, channels and measurements defined in the PulseTemplate can be renamed/mapped
        via the channel_mapping and measurement_mapping arguments.

        :param parameters: A mapping of parameter names to Parameter objects.
        :param measurement_mapping: A mapping of measurement window names. Windows that are mapped to None are omitted.
        :param channel_mapping: A mapping of channel names. Channels that are mapped to None are omitted.
        :param global_transformation: This transformation is applied to every waveform
        :param to_single_waveform: A set of pulse templates (or identifiers) which are directly translated to a
        waveform. This might change how transformations are applied. TODO: clarify
        :return: A Loop object corresponding to this PulseTemplate.
        """
        if parameters is None:
            parameters = dict()
        if measurement_mapping is None:
            measurement_mapping = {name: name for name in self.measurement_names}
        if channel_mapping is None:
            channel_mapping = dict()
        if to_single_waveform is None:
            to_single_waveform = set()

        # make sure all values in the parameters dict are of type Parameter
        for (key, value) in parameters.items():
            if not isinstance(value, Parameter):
                parameters[key] = ConstantParameter(value)

        root_loop = Loop()
        # call subclass specific implementation
        self._create_program(parameters=parameters,
                             measurement_mapping=measurement_mapping,
                             channel_mapping=channel_mapping,
                             global_transformation=global_transformation,
                             to_single_waveform=to_single_waveform,
                             parent_loop=root_loop)

        if root_loop.waveform is None and len(root_loop.children) == 0:
            return None # return None if no program
        return root_loop

    @abstractmethod
    def _internal_create_program(self, *,
                                 parameters: Dict[str, Parameter],
                                 measurement_mapping: Dict[str, Optional[str]],
                                 channel_mapping: Dict[ChannelID, Optional[ChannelID]],
                                 global_transformation: Optional[Transformation],
                                 to_single_waveform: Set[Union[str, 'PulseTemplate']],
                                 parent_loop: Loop) -> None:
        """The subclass specific implementation of create_program().

        Receives a Loop instance parent_loop to which it should append measurements and its own Loops as children.

        Subclasses should not overwrite create_program() directly but provide their implementation here. This method
        is called by create_program().
        Implementations should not call create_program() of any subtemplates to obtain Loop objects for them but
        call subtemplate._internal_create_program() instead, providing an adequate parent_loop object to which
        the subtemplate will append. Implementations must make sure not to append invalid Loop objects (no waveform or no children).

        In case of an error (e.g. invalid measurement mapping, missing parameters, violated parameter constraints, etc),
        implementations of this method must throw an adequate exception. They do not have to ensure that the parent_loop
        remains unchanged in this case."""

    def _create_program(self, *,
                        parameters: Dict[str, Parameter],
                        measurement_mapping: Dict[str, Optional[str]],
                        channel_mapping: Dict[ChannelID, Optional[ChannelID]],
                        global_transformation: Optional[Transformation],
                        to_single_waveform: Set[Union[str, 'PulseTemplate']],
                        parent_loop: Loop):
        """Generic part of create program. This method handles to_single_waveform and the configuration of the
        transformer."""
        if self.identifier in to_single_waveform or self in to_single_waveform:
            root = Loop()

            self._internal_create_program(parameters=parameters,
                                          measurement_mapping=measurement_mapping,
                                          channel_mapping=channel_mapping,
                                          global_transformation=None,
                                          to_single_waveform=to_single_waveform,
                                          parent_loop=root)

            waveform = to_waveform(root)

            if global_transformation:
                waveform = TransformingWaveform(waveform, global_transformation)

            # convert the nicely formatted measurement windows back into the old format again :(
            measurements = root.get_measurement_windows()
            measurement_window_list = []
            for measurement_name, (begins, lengths) in measurements.items():
                measurement_window_list.extend(zip(itertools.repeat(measurement_name), begins, lengths))

            parent_loop.add_measurements(measurement_window_list)
            parent_loop.append_child(waveform=waveform)

        else:
            self._internal_create_program(parameters=parameters,
                                          measurement_mapping=measurement_mapping,
                                          channel_mapping=channel_mapping,
                                          to_single_waveform=to_single_waveform,
                                          global_transformation=global_transformation,
                                          parent_loop=parent_loop)


class AtomicPulseTemplate(PulseTemplate, MeasurementDefiner):
    """A PulseTemplate that does not imply any control flow disruptions and can be directly
    translated into a waveform.

    Implies that no AtomicPulseTemplate object is interruptable.
    """
    def __init__(self, *,
                 identifier: Optional[str],
                 measurements: Optional[List[MeasurementDeclaration]]):
        PulseTemplate.__init__(self, identifier=identifier)
        MeasurementDefiner.__init__(self, measurements=measurements)

    def is_interruptable(self) -> bool:
        return False

    @property
    def atomicity(self) -> bool:
        return True

    measurement_names = MeasurementDefiner.measurement_names

    def build_sequence(self,
                       sequencer: Sequencer,
                       parameters: Dict[str, Parameter],
                       conditions: Dict[str, Condition],
                       measurement_mapping: Dict[str, Optional[str]],
                       channel_mapping: Dict[ChannelID, Optional[ChannelID]],
                       instruction_block: InstructionBlock) -> None:
        parameters = {parameter_name: parameter_value.get_value()
                      for parameter_name, parameter_value in parameters.items()
                      if parameter_name in self.parameter_names}
        waveform = self.build_waveform(parameters,
                                       channel_mapping=channel_mapping)
        if waveform:
            measurements = self.get_measurement_windows(parameters=parameters, measurement_mapping=measurement_mapping)
            instruction_block.add_instruction_meas(measurements)
            instruction_block.add_instruction_exec(waveform)

    def _internal_create_program(self, *,
                                 parameters: Dict[str, Parameter],
                                 measurement_mapping: Dict[str, Optional[str]],
                                 channel_mapping: Dict[ChannelID, Optional[ChannelID]],
                                 global_transformation: Optional[Transformation],
                                 to_single_waveform: Set[Union[str, 'PulseTemplate']],
                                 parent_loop: Loop) -> None:
        """Parameter constraints are validated in build_waveform because build_waveform is guaranteed to be called
        during sequencing"""
        ### current behavior (same as previously): only adds EXEC Loop and measurements if a waveform exists.
        ### measurements are directly added to parent_loop (to reflect behavior of Sequencer + MultiChannelProgram)
        # todo (2018-08-08): could move measurements into own Loop object?

        # todo (2018-07-05): why are parameter constraints not validated here?
        try:
            parameters = {parameter_name: parameters[parameter_name].get_value()
                          for parameter_name in self.parameter_names}

            measurement_parameters = {parameter_name: parameters[parameter_name].get_value()
                                      for parameter_name in self.measurement_parameters}
        except KeyError as e:
            raise ParameterNotProvidedException(str(e)) from e

        measurements = self.get_measurement_windows(parameters=measurement_parameters, measurement_mapping=measurement_mapping)
        waveform = self.build_waveform(parameters=parameters,
                                       channel_mapping=channel_mapping)
        if waveform:
            if global_transformation:
                waveform = TransformingWaveform(waveform, global_transformation)

            parent_loop.add_measurements(measurements=measurements)
            parent_loop.append_child(waveform=waveform)

    @abstractmethod
    def build_waveform(self,
                       parameters: Dict[str, Real],
                       channel_mapping: Dict[ChannelID, Optional[ChannelID]]) -> Optional[Waveform]:
        """Translate this PulseTemplate into a waveform according to the given parameters.


        Subclasses of AtomicPulseTemplate must check for ParameterConstraintViolation
        errors in their build_waveform implementation and raise corresponding exceptions.

        Args:
            parameters (Dict(str -> Parameter)): A mapping of parameter names to real numbers.
            channel_mapping (Dict(ChannelID -> ChannelID): A mapping of Channel IDs
        Returns:
            Waveform object represented by this PulseTemplate object or None, if this object
                does not represent a valid waveform of finite length.
        """


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

