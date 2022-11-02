"""This module defines the abstract PulseTemplate class which is the basis of any
pulse model in the qupulse.

Classes:
    - PulseTemplate: Represents the parametrized general structure of a pulse.
    - AtomicPulseTemplate: PulseTemplate that does imply any control flow disruptions and can be
        directly translated into a waveform.
"""
from abc import abstractmethod
from typing import Dict, Tuple, Set, Optional, Union, List, Callable, Any, Generic, TypeVar, Mapping
import itertools
import collections
from numbers import Real, Number

import sympy

from qupulse.utils.types import ChannelID, DocStringABCMeta, FrozenDict
from qupulse.utils import forced_hash
from qupulse.serialization import Serializable
from qupulse.expressions import ExpressionScalar, Expression, ExpressionLike
from qupulse._program._loop import Loop, to_waveform
from qupulse._program.transformation import Transformation, IdentityTransformation, ChainedTransformation, chain_transformations

from qupulse.pulses.parameters import Parameter, ConstantParameter, ParameterNotProvidedException
from qupulse._program.waveforms import Waveform, TransformingWaveform
from qupulse.pulses.measurement import MeasurementDefiner, MeasurementDeclaration
from qupulse.parameter_scope import Scope, DictScope

__all__ = ["PulseTemplate", "AtomicPulseTemplate", "DoubleParameterNameException", "MappingTuple"]


MappingTuple = Union[Tuple['PulseTemplate'],
                     Tuple['PulseTemplate', Dict],
                     Tuple['PulseTemplate', Dict, Dict],
                     Tuple['PulseTemplate', Dict, Dict, Dict]]


class PulseTemplate(Serializable):
    """A PulseTemplate represents the parametrized general structure of a pulse.

    A PulseTemplate described a pulse in an abstract way: It defines the structure of a pulse
    but might leave some timings or voltage levels undefined, thus declaring parameters.
    This allows to reuse a PulseTemplate for several pulses which have the same overall structure
    and differ only in concrete values for the parameters.
    Obtaining an actual pulse which can be executed by specifying values for these parameters is
    called instantiation of the PulseTemplate and achieved by invoking the sequencing process.
    """

    """This is not stable"""
    _DEFAULT_FORMAT_SPEC = 'identifier'

    def __init__(self, *,
                 identifier: Optional[str]) -> None:
        super().__init__(identifier=identifier)
        self.__cached_hash_value = None

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
        from qupulse.pulses.sequence_pulse_template import SequencePulseTemplate

        return SequencePulseTemplate.concatenate(self, other)

    def __rmatmul__(self, other: MappingTuple) -> 'SequencePulseTemplate':
        from qupulse.pulses.sequence_pulse_template import SequencePulseTemplate

        return SequencePulseTemplate.concatenate(other, self)

    @property
    @abstractmethod
    def integral(self) -> Dict[ChannelID, ExpressionScalar]:
        """Returns an expression giving the integral over the pulse."""

    @property
    def initial_values(self) -> Dict[ChannelID, ExpressionScalar]:
        """Values of defined channels at t == 0"""
        raise NotImplementedError(f"The pulse template of type {type(self)} does not implement `initial_values`")

    @property
    def final_values(self) -> Dict[ChannelID, ExpressionScalar]:
        """Values of defined channels at t == self.duration"""
        raise NotImplementedError(f"The pulse template of type {type(self)} does not implement `final_values`")

    def create_program(self, *,
                       parameters: Optional[Mapping[str, Union[Expression, str, Number, ConstantParameter]]]=None,
                       measurement_mapping: Optional[Mapping[str, Optional[str]]]=None,
                       channel_mapping: Optional[Mapping[ChannelID, Optional[ChannelID]]]=None,
                       global_transformation: Optional[Transformation]=None,
                       to_single_waveform: Set[Union[str, 'PulseTemplate']]=None,
                       volatile: Set[str] = None) -> Optional['Loop']:
        """Translates this PulseTemplate into a program Loop.

        The returned Loop represents the PulseTemplate with all parameter values instantiated provided as dictated by
        the parameters argument. Optionally, channels and measurements defined in the PulseTemplate can be renamed/mapped
        via the channel_mapping and measurement_mapping arguments.

        Args:
            parameters: A mapping of parameter names to Parameter objects.
            measurement_mapping: A mapping of measurement window names. Windows that are mapped to None are omitted.
            channel_mapping: A mapping of channel names. Channels that are mapped to None are omitted.
            global_transformation: This transformation is applied to every waveform
            to_single_waveform: A set of pulse templates (or identifiers) which are directly translated to a
                waveform. This might change how transformations are applied. TODO: clarify
            volatile: Everything in the final program that depends on these parameters is marked as volatile
        Returns:
             A Loop object corresponding to this PulseTemplate.
        """
        if parameters is None:
            parameters = dict()
        if measurement_mapping is None:
            measurement_mapping = {name: name for name in self.measurement_names}
        if channel_mapping is None:
            channel_mapping = dict()
        if to_single_waveform is None:
            to_single_waveform = set()
        if volatile is None:
            volatile = set()

        # make sure all channels are mapped
        complete_channel_mapping = {channel: channel for channel in self.defined_channels}
        complete_channel_mapping.update(channel_mapping)

        non_unique_targets = {channel
                              for channel, count in collections.Counter(channel_mapping.values()).items()
                              if count > 1 and channel is not None}
        if non_unique_targets:
            raise ValueError('The following channels are mapped to twice', non_unique_targets)

        # make sure all values in the parameters dict are numbers
        if isinstance(parameters, Scope):
            assert not volatile
            scope = parameters
        else:
            parameters = dict(parameters)
            for parameter_name, value in parameters.items():
                if isinstance(value, Parameter):
                    parameters[parameter_name] = value.get_value()
                elif not isinstance(value, Number):
                    parameters[parameter_name] = Expression(value).evaluate_numeric()

            scope = DictScope(values=FrozenDict(parameters), volatile=volatile)

        root_loop = Loop()

        # call subclass specific implementation
        self._create_program(scope=scope,
                             measurement_mapping=measurement_mapping,
                             channel_mapping=complete_channel_mapping,
                             global_transformation=global_transformation,
                             to_single_waveform=to_single_waveform,
                             parent_loop=root_loop)

        if root_loop.waveform is None and len(root_loop.children) == 0:
            return None  # return None if no program
        return root_loop

    @abstractmethod
    def _internal_create_program(self, *,
                                 scope: Scope,
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
                        scope: Scope,
                        measurement_mapping: Dict[str, Optional[str]],
                        channel_mapping: Dict[ChannelID, Optional[ChannelID]],
                        global_transformation: Optional[Transformation],
                        to_single_waveform: Set[Union[str, 'PulseTemplate']],
                        parent_loop: Loop):
        """Generic part of create program. This method handles to_single_waveform and the configuration of the
        transformer."""
        if self.identifier in to_single_waveform or self in to_single_waveform:
            root = Loop()

            if not scope.get_volatile_parameters().keys().isdisjoint(self.parameter_names):
                raise NotImplementedError('A pulse template that has volatile parameters cannot be transformed into a '
                                          'single waveform yet.')

            self._internal_create_program(scope=scope,
                                          measurement_mapping=measurement_mapping,
                                          channel_mapping=channel_mapping,
                                          global_transformation=None,
                                          to_single_waveform=to_single_waveform,
                                          parent_loop=root)

            waveform = to_waveform(root)

            if global_transformation:
                waveform = TransformingWaveform.from_transformation(waveform, global_transformation)

            # convert the nicely formatted measurement windows back into the old format again :(
            measurements = root.get_measurement_windows()
            measurement_window_list = []
            for measurement_name, (begins, lengths) in measurements.items():
                measurement_window_list.extend(zip(itertools.repeat(measurement_name), begins, lengths))

            parent_loop.add_measurements(measurement_window_list)
            parent_loop.append_child(waveform=waveform)

        else:
            self._internal_create_program(scope=scope,
                                          measurement_mapping=measurement_mapping,
                                          channel_mapping=channel_mapping,
                                          to_single_waveform=to_single_waveform,
                                          global_transformation=global_transformation,
                                          parent_loop=parent_loop)

    def __format__(self, format_spec: str):
        if format_spec == '':
            format_spec = self._DEFAULT_FORMAT_SPEC
        formatted = []
        for attr in format_spec.split(';'):
            value = getattr(self, attr)
            if value is None:
                continue
            # the repr(str(value)) is to avoid very deep nesting. If needed one should use repr
            formatted.append('{attr}={value}'.format(attr=attr, value=repr(str(value))))
        type_name = type(self).__name__
        return '{type_name}({attrs})'.format(type_name=type_name, attrs=', '.join(formatted))

    def __str__(self):
        return format(self)

    def __repr__(self):
        type_name = type(self).__name__
        kwargs = ','.join('%s=%r' % (key, value)
                          for key, value in self.get_serialization_data().items()
                          if key.isidentifier() and value is not None)
        return '{type_name}({kwargs})'.format(type_name=type_name, kwargs=kwargs)

    def __add__(self, other: ExpressionLike):
        from qupulse.pulses.arithmetic_pulse_template import try_operation
        return try_operation(self, '+', other)

    def __radd__(self, other: ExpressionLike):
        from qupulse.pulses.arithmetic_pulse_template import try_operation
        return try_operation(other, '+', self)

    def __sub__(self, other):
        from qupulse.pulses.arithmetic_pulse_template import try_operation
        return try_operation(self, '-', other)

    def __rsub__(self, other):
        from qupulse.pulses.arithmetic_pulse_template import try_operation
        return try_operation(other, '-', self)

    def __mul__(self, other):
        from qupulse.pulses.arithmetic_pulse_template import try_operation
        return try_operation(self, '*', other)

    def __rmul__(self, other):
        from qupulse.pulses.arithmetic_pulse_template import try_operation
        return try_operation(other, '*', self)

    def __truediv__(self, other):
        from qupulse.pulses.arithmetic_pulse_template import try_operation
        return try_operation(self, '/', other)

    def __hash__(self):
        if self.__cached_hash_value is None:
            self.__cached_hash_value = forced_hash(self.get_serialization_data())
        return self.__cached_hash_value


class AtomicPulseTemplate(PulseTemplate, MeasurementDefiner):
    """A PulseTemplate that does not imply any control flow disruptions and can be directly
    translated into a waveform.

    Implies that no AtomicPulseTemplate object is interruptable.
    """
    _AS_EXPRESSION_TIME = sympy.Dummy('_t', positive=True)

    def __init__(self, *,
                 identifier: Optional[str],
                 measurements: Optional[List[MeasurementDeclaration]]):
        PulseTemplate.__init__(self, identifier=identifier)
        MeasurementDefiner.__init__(self, measurements=measurements)

    @property
    def atomicity(self) -> bool:
        return True

    measurement_names = MeasurementDefiner.measurement_names

    def _internal_create_program(self, *,
                                 scope: Scope,
                                 measurement_mapping: Dict[str, Optional[str]],
                                 channel_mapping: Dict[ChannelID, Optional[ChannelID]],
                                 global_transformation: Optional[Transformation],
                                 to_single_waveform: Set[Union[str, 'PulseTemplate']],
                                 parent_loop: Loop) -> None:
        """Parameter constraints are validated in build_waveform because build_waveform is guaranteed to be called
        during sequencing"""
        ### current behavior (same as previously): only adds EXEC Loop and measurements if a waveform exists.
        ### measurements are directly added to parent_loop (to reflect behavior of Sequencer + MultiChannelProgram)
        assert not scope.get_volatile_parameters().keys() & self.parameter_names, "AtomicPT cannot be volatile"

        waveform = self.build_waveform(parameters=scope,
                                       channel_mapping=channel_mapping)
        if waveform:
            measurements = self.get_measurement_windows(parameters=scope,
                                                        measurement_mapping=measurement_mapping)

            if global_transformation:
                waveform = TransformingWaveform.from_transformation(waveform, global_transformation)

            parent_loop.add_measurements(measurements=measurements)
            parent_loop.append_child(waveform=waveform)

    @abstractmethod
    def build_waveform(self,
                       parameters: Mapping[str, Real],
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

    def _as_expression(self) -> Dict[ChannelID, ExpressionScalar]:
        """Helper function to allow integral calculation in case of truncation. AtomicPulseTemplate._AS_EXPRESSION_TIME
        is by convention the time variable."""
        raise NotImplementedError(f"_as_expression is not implemented for {type(self)} "
                                  f"which means it cannot be truncated and integrated over.")

    @property
    def integral(self) -> Dict[ChannelID, ExpressionScalar]:
        # this default implementation uses _as_expression
        return {ch: ExpressionScalar(sympy.integrate(expr.sympified_expression,
                                                     (self._AS_EXPRESSION_TIME, 0, self.duration.sympified_expression)))
                for ch, expr in self._as_expression().items()}

    @property
    def initial_values(self) -> Dict[ChannelID, ExpressionScalar]:
        values = self._as_expression()
        for ch, value in values.items():
            values[ch] = value.evaluate_symbolic({self._AS_EXPRESSION_TIME: 0})
        return values

    @property
    def final_values(self) -> Dict[ChannelID, ExpressionScalar]:
        values = self._as_expression()
        for ch, value in values.items():
            values[ch] = value.evaluate_symbolic({self._AS_EXPRESSION_TIME: self.duration})
        return values


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

