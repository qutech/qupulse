# SPDX-FileCopyrightText: 2014-2024 Quantum Technology Group and Chair of Software Engineering, RWTH Aachen University
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""This module defines the abstract PulseTemplate class which is the basis of any
pulse model in the qupulse.

Classes:
    - PulseTemplate: Represents the parametrized general structure of a pulse.
    - AtomicPulseTemplate: PulseTemplate that does imply any control flow disruptions and can be
        directly translated into a waveform.
"""
import warnings
from abc import abstractmethod
from typing import Dict, Tuple, Set, Optional, Union, List, Callable, Any, Mapping, Literal
import collections
from numbers import Real, Number

import numpy
import sympy

from qupulse.pulses.metadata import TemplateMetadata, SingleWaveformStrategy
from qupulse.utils.types import ChannelID, FrozenDict
from qupulse.utils import forced_hash
from qupulse.serialization import Serializable
from qupulse.expressions import ExpressionScalar, Expression, ExpressionLike
from qupulse.program.transformation import Transformation

from qupulse.program.waveforms import Waveform, TransformingWaveform, WaveformMetadata
from qupulse.pulses.measurement import MeasurementDefiner, MeasurementDeclaration
from qupulse.parameter_scope import Scope, DictScope

from qupulse.program import ProgramBuilder, default_program_builder, Program

__all__ = ["PulseTemplate", "AtomicPulseTemplate", "DoubleParameterNameException", "MappingTuple",
           "UnknownVolatileParameter"]


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

    _CAST_INT_TO_INT64 = True

    def __init__(self, *,
                 identifier: Optional[str],
                 metadata: Union[TemplateMetadata, dict] = None) -> None:
        super().__init__(identifier=identifier)
        if isinstance(metadata, dict):
            metadata = TemplateMetadata(**metadata)
        self._metadata = metadata

        self.__cached_hash_value = None

    @property
    def metadata(self) -> TemplateMetadata:
        """The metadata is intended for information which does not concern the pulse itself but rather its usage.

        Here is the place for program builder optimization hints or tags that are not targeted at qupulse. Currently,
        qupulse itself only uses the :py:attr:`.TemplateMetadata.to_single_waveform` attribute to allow dynamic atomicity during translation.

        As it is no property of the pulse itself, it is ignored for hashing and equality checks.
        """
        if (metadata := self._metadata) is None:
            self._metadata = metadata = TemplateMetadata()
        return metadata

    def get_serialization_data(self, serializer: Optional['Serializer'] = None) -> Dict[str, Any]:
        data = super().get_serialization_data(serializer=serializer)
        if self._metadata is not None and (metadata := self._metadata.get_serialization_data()):
            data["metadata"] = metadata
        return data

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

    def _is_atomic(self) -> bool:
        """This is (currently a private) a check if this pulse template always is translated into a single waveform."""
        return self.metadata.to_single_waveform == 'always'

    @property
    def to_single_waveform(self) -> Optional[SingleWaveformStrategy]:
        """This property describes whether this pulse template is translated into a single waveform.

        'always': It is always translated into a single waveform.
        None: It depends on the `create_program` arguments and the pulse template itself.
        """
        warnings.warn("to_single_waveform is deprecated. Use metadata.to_single_waveform instead.",
                      category=DeprecationWarning, stacklevel=2)
        return self.metadata.to_single_waveform

    def __matmul__(self, other: Union['PulseTemplate', MappingTuple]) -> 'SequencePulseTemplate':
        """This method enables using the @-operator (intended for matrix multiplication) for
         concatenating pulses. If one of the pulses is a SequencePulseTemplate the other pulse gets merged into it"""
        from qupulse.pulses.sequence_pulse_template import SequencePulseTemplate

        return SequencePulseTemplate.concatenate(self, other)

    def __rmatmul__(self, other: MappingTuple) -> 'SequencePulseTemplate':
        from qupulse.pulses.sequence_pulse_template import SequencePulseTemplate

        return SequencePulseTemplate.concatenate(other, self)

    def __pow__(self, power: ExpressionLike):
        """This is a convenience wrapper for :func:`.with_repetition`."""
        return self.with_repetition(power)

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
                       parameters: Optional[Mapping[str, Union[Expression, str, Number]]]=None,
                       measurement_mapping: Optional[Mapping[str, Optional[str]]]=None,
                       channel_mapping: Optional[Mapping[ChannelID, Optional[ChannelID]]]=None,
                       global_transformation: Optional[Transformation]=None,
                       to_single_waveform: Set[str]=None,
                       volatile: Union[Set[str], str] = None,
                       program_builder: ProgramBuilder = None) -> Optional[Program]:
        """Translates this PulseTemplate into a program Loop.

        The returned Loop represents the PulseTemplate with all parameter values instantiated provided as dictated by
        the parameters argument. Optionally, channels and measurements defined in the PulseTemplate can be renamed/mapped
        via the channel_mapping and measurement_mapping arguments.

        Args:
            parameters: A mapping of parameter names to Parameter objects.
            measurement_mapping: A mapping of measurement window names. Windows that are mapped to None are omitted.
            channel_mapping: A mapping of channel names. Channels that are mapped to None are omitted.
            global_transformation: This transformation is applied to every waveform
            to_single_waveform: A set of identifiers which are directly translated to a waveform.
            volatile: Everything in the final program that depends on these parameters is marked as volatile
            program_builder: This program builder is used to build the return value. If `None` `default_program_builder`
                is used.
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
        elif not all(isinstance(elem, str) for elem in to_single_waveform):
            non_str_types = {type(elem).__name__ for elem in to_single_waveform if not isinstance(elem, str)}
            warnings.warn(f"Non str to_single_waveform members are ignored: {non_str_types}",
                          category=NonStrToSingleWaveformWarning, stacklevel=2)
            to_single_waveform = {elem for elem in to_single_waveform if isinstance(elem, str)}
        if volatile is None:
            volatile = set()
        elif isinstance(volatile, str):
            volatile = {volatile}
        else:
            volatile = set(volatile)
        if program_builder is None:
            program_builder = default_program_builder()

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
            to_int = numpy.int64 if self._CAST_INT_TO_INT64 else lambda x: x
            for parameter_name, value in parameters.items():
                if type(value) is int:
                    # numpy casts ints to int32 per default on windows
                    # this can easily lead to overflows when times of the order of seconds
                    # are represented with integers
                    parameters[parameter_name] = to_int(value)

                elif not isinstance(value, Number):
                    parameters[parameter_name] = Expression(value).evaluate_numeric()

            scope = DictScope(values=FrozenDict(parameters), volatile=volatile)

        for volatile_name in scope.get_volatile_parameters():
            if volatile_name not in scope:
                warnings.warn(f"The volatile parameter {volatile_name!r} is not in the given parameters.",
                              category=UnknownVolatileParameter,
                              stacklevel=2)

        program_builder.override(
            scope=scope,
            measurement_mapping=measurement_mapping,
            channel_mapping=complete_channel_mapping,
            global_transformation=global_transformation,
            to_single_waveform=to_single_waveform,
        )

        # call subclass specific implementation
        self._build_program(program_builder=program_builder)
        return program_builder.to_program()

    def _build_program(self, program_builder: ProgramBuilder):
        """New method that keeps state in program builder"""
        if (validate_scope := getattr(self, "validate_scope", None)) is not None:
            validate_scope(program_builder.build_context.scope)

        with program_builder.with_metadata(self.metadata, self.identifier) as inner_program_builder:
            self._internal_build_program(inner_program_builder)

    def _internal_build_program(self, program_builder: ProgramBuilder):
        """The subclass specific implementation of create_program()."""
        context = program_builder.build_context
        settings = program_builder.build_settings
        self._internal_create_program(
            scope=context.scope,
            measurement_mapping=dict(context.measurement_mapping),
            channel_mapping=dict(context.channel_mapping),
            global_transformation=context.transformation,
            to_single_waveform=set(settings.to_single_waveform),
            program_builder=program_builder
        )

    def _create_program(self, *,
                        scope: Scope,
                        measurement_mapping: Dict[str, Optional[str]],
                        channel_mapping: Dict[ChannelID, Optional[ChannelID]],
                        global_transformation: Optional[Transformation],
                        to_single_waveform: Set[Union[str, 'PulseTemplate']],
                        program_builder: ProgramBuilder):
        """Generic part of create program. This method handles to_single_waveform and the configuration of the
        transformer."""
        warnings.warn("_create_program is deprecated. "
                      "Update your driver to use _build_program",
                      DeprecationWarning, stacklevel=2)
        if self.metadata.to_single_waveform == 'always' or self.identifier in to_single_waveform or self in to_single_waveform:
            with program_builder.new_subprogram(global_transformation=global_transformation) as inner_program_builder:

                if not scope.get_volatile_parameters().keys().isdisjoint(self.parameter_names):
                    raise NotImplementedError('A pulse template that has volatile parameters cannot be transformed into a '
                                              'single waveform yet.')

                self._internal_create_program(scope=scope,
                                              measurement_mapping=measurement_mapping,
                                              channel_mapping=channel_mapping,
                                              global_transformation=None,
                                              to_single_waveform=to_single_waveform,
                                              program_builder=inner_program_builder)

        else:
            self._internal_create_program(scope=scope,
                                          measurement_mapping=measurement_mapping,
                                          channel_mapping=channel_mapping,
                                          to_single_waveform=to_single_waveform,
                                          global_transformation=global_transformation,
                                          program_builder=program_builder)

    def with_parallel_channels(self, values: Mapping[ChannelID, ExpressionLike]) -> 'PulseTemplate':
        """Create a new pulse template that sets the given channels to the corresponding values.

        See :class:`~qupulse.pulses.ParallelChannelPulseTemplate` for implementation details and restictions.

        Examples:
            >>> from qupulse.pulses import FunctionPT
            ... fpt = FunctionPT('sin(0.1 * t)', duration_expression=10)
            ... fpt_and_marker = fpt.with_parallel_channels({'marker': 1})

        Args:
            values: Values to be set for each channel.

        Returns:
            A newly created pulse template.
        """
        from qupulse.pulses.multi_channel_pulse_template import ParallelChannelPulseTemplate
        return ParallelChannelPulseTemplate(
            self,
            values
        )

    def with_repetition(self, repetition_count: ExpressionLike) -> 'PulseTemplate':
        """Repeat this pulse template `repetition_count` times via a :class:`~qupulse.pulses.RepetitionPulseTemplate`.

        Examples:
            >>> from qupulse.pulses import FunctionPT
            ... fpt = FunctionPT('sin(0.1 * t)', duration_expression=10)
            ... repeated = fpt.with_repetition('n_periods')

        Args:
            repetition_count: Amount of times this pulse template is repeated in the return value.

        Returns:
            A newly created pulse template.
        """
        from qupulse.pulses.repetition_pulse_template import RepetitionPulseTemplate
        return RepetitionPulseTemplate(self, repetition_count)

    def with_mapping(self, *mapping_tuple_args: Mapping, **mapping_kwargs: Mapping) -> 'PulseTemplate':
        """Map parameters / channel names / measurement names. You may either specify the mappings as positional
        arguments XOR as keyword arguments. Positional arguments are forwarded to
        :func:`~qupulse.pulses.MappingPT.from_tuple` which automatically determines the "type" of the mappings.
        Keyword arguments must be one of the keyword arguments of :class:`~qupulse.pulses.MappingPT`.

        Args:
            *mapping_tuple_args: Mappings for parameters / channel names / measurement names
            **mapping_kwargs: Mappings for parameters / channel names / measurement names

        Examples:
            Equivalent ways to rename a channel and map a parameter value
            >>> from qupulse.pulses import FunctionPT
            ... fpt = FunctionPT('sin(f * t)', duration_expression=10, channel='A')
            ... mapped = fpt.with_mapping({'f': 0.1}, {'A': 'B'})
            ... mapped.defined_channels
            {'B'}

            >>> from qupulse.pulses import FunctionPT
            ... fpt = FunctionPT('sin(f * t)', duration_expression=10, channel='A')
            ... mapped = fpt.with_mapping(parameter_mapping={'f': 0.1}, channel_mapping={'A': 'B'})
            ... mapped.defined_channels
            {'B'}

        Returns:
            A newly created mapping pulse template
        """
        from qupulse.pulses import MappingPT

        if mapping_kwargs and mapping_tuple_args:
            raise ValueError("Only positional argument (auto detection of mapping type) "
                             "xor keyword arguments are allowed.")
        if mapping_tuple_args:
            return MappingPT.from_tuple((self, *mapping_tuple_args))
        else:
            return MappingPT(self, **mapping_kwargs)

    def with_iteration(self, loop_idx: str, loop_range) -> 'PulseTemplate':
        """Create a :class:`~qupulse.pulses.ForLoopPT` with the given index and range.

        Examples:
            >>> from qupulse.pulses import ConstantPT
            ... const = ConstantPT('t_hold', {'x': 'start_x + i_x * step_x', 'y': 'start_y + i_y * step_y'})
            ... scan_2d = const.with_iteration('i_x', 'n_x').with_iteration('i_y', 'n_y')
        """
        from qupulse.pulses import ForLoopPT
        return ForLoopPT(self, loop_idx, loop_range)

    def with_time_reversal(self) -> 'PulseTemplate':
        """Reverse this pulse template by creating a :class:`~qupulse.pulses.TimeReversalPT`.

        Examples:
            >>> from qupulse.pulses import FunctionPT
            ... forward = FunctionPT('sin(f * t)', duration_expression=10, channel='A')
            ... backward = fpt.with_time_reversal()
            ... forward_and_backward = forward @ backward
        """
        from qupulse.pulses import TimeReversalPT
        return TimeReversalPT(self)

    def with_appended(self, *appended: 'PulseTemplate'):
        """Create a :class:`~qupulse.pulses.SequencePT` that represents a sequence of this pulse template and `appended`

        You can also use the `@` operator to do this or call :func:`qupulse.pulses.SequencePT.concatenate` directly.
        """
        from qupulse.pulses import SequencePT
        if appended:
            return SequencePT.concatenate(self, *appended)
        else:
            return self

    def with_mapped_subtemplates(self,
                                 map_fn: Callable[['PulseTemplate'], 'PulseTemplate'], *,
                                 identifier_map: Callable[[Optional[str]], Optional[str]] = lambda x: x,
                                 always_new_template: bool = False,
                                 recursion_strategy: Literal['pre', 'post', 'self'] = 'pre') -> 'PulseTemplate':
        """Return a new pulse template with all subtemplates mapped by ``map_fn``.

        If ``map_fn`` returns the same object for all subtemplates no new pulse template is created unless ``always_new_template`` is true.

        If a new template is created the identifier of the old template is passed to ``ìdentifier_map`` to determine the new identifier. It is the same by default.

        By default, this function visits all subtemplates recursively. The ``recursion_strategy`` determines the order of recursion and ``map_fn`` calls.
        For the default 'pre' the recursive subtemplates are mapped first before the direct subtemplates are mapped.
        For 'post' the direct subtemplates are mapped first before the recursive subtemplates are visited.
        For 'self' there is no recursion, i.e. the potential recursion can be implemented in ``map_fn`` itself.

        This helper function is useful for modification of pulse templates without having to worry about the internal
        structure.

        The pulse registry feature is not yet supported by the method. When you need it, please open an issue on github.

        Args:
            map_fn: The function to be applied to the subtemplates according to the recursion strategy.
            identifier_map: This function is called to map the identifiers of pulse templates that need to be newly created because their subtemplates changed.
            always_new_template: If True, a new pulse template will be created even if the mapped subtemplates are identical.
            recursion_strategy: Either 'pre', 'post' or 'self'.
                                 - 'pre': All recursive subtemplates are mapped before the map_fn is applied.
                                 - 'post': All recursive subtemplates are mapped after the map_fn is applied.
                                 - 'Self': The recursion needs to be done by the map_fn if required.

        Returns:
            A pulse template with mapped subtemplates.
        """
        assert recursion_strategy in ('pre', 'post', 'self')

        def map_templates_in_tree(tree):
            if isinstance(tree, PulseTemplate):
                pt = tree
                if recursion_strategy == 'pre':
                    pt = pt.with_mapped_subtemplates(map_fn,
                                                     identifier_map=identifier_map,
                                                     always_new_template=always_new_template,
                                                     recursion_strategy=recursion_strategy,
                                                     )
                pt = map_fn(pt)
                if recursion_strategy == 'post':
                    pt = pt.with_mapped_subtemplates(map_fn,
                                                     identifier_map=identifier_map,
                                                     always_new_template=always_new_template,
                                                     recursion_strategy=recursion_strategy,
                                                     )
                map_templates_in_tree.tree_changed |= pt is not tree
                return pt

            elif isinstance(tree, tuple):
                return tuple(map(map_templates_in_tree, tree))
            elif isinstance(tree, list):
                return list(map(map_templates_in_tree, tree))
            elif isinstance(tree, dict):
                return {key: map_templates_in_tree(value)
                        for key, value in tree.items()}
            else:
                return tree
        map_templates_in_tree.tree_changed = always_new_template

        data = {name: value
                for name, value in self.get_serialization_data().items()
                if not name.startswith('#')}
        mapped = map_templates_in_tree(data)
        if map_templates_in_tree.tree_changed:
            identifier = identifier_map(self.identifier)
            return self.deserialize(**mapped, identifier=identifier)
        else:
            return self

    def pad_to(self,
               to_new_duration: Union[ExpressionLike, Callable[[Expression], ExpressionLike]],
               spt_kwargs: Mapping[str, Any] = None,
               pt_kwargs: Mapping[str, Any] = None,
               ) -> 'PulseTemplate':
        """Pad this pulse template to the given duration.
        The target duration can be numeric, symbolic or a callable that returns a new duration from the current
        duration.

        Examples:
            # pad to a fixed duration
            >>> padded_1 = my_pt.pad_to(1000)

            # pad to a fixed sample coun
            >>> padded_2 = my_pt.pad_to('sample_rate * 1000')

            # pad to the next muliple of 16 samples with a symbolic sample rate
            >>> padded_3 = my_pt.pad_to(to_next_multiple('sample_rate', 16))

            # pad to the next muliple of 16 samples with a fixed sample rate of 1 GHz
            >>> padded_4 = my_pt.pad_to(to_next_multiple(1, 16))
        Args:
            to_new_duration: Duration or callable that maps the current duration to the new duration
            spt_kwargs: Keyword arguments for an optionally newly created sequence pulse template.
            pt_kwargs: Deprecated! Similar to ``spt_kwargs`` but enforces sequence pt creation even if no padding required.

        Returns:
            A pulse template that has the duration given by ``to_new_duration``. It can be ``self`` if the duration is
            already as required. It is never ``self`` if ``pt_kwargs`` is non-empty (deprecated feated).
        """
        if pt_kwargs:
            warnings.warn('pt_kwargs is deprecated and will be removed in a post 1.0 release. '
                          'Please use spt_kwargs instead.',
                          DeprecationWarning, stacklevel=2)
        spt_kwargs = spt_kwargs or {}
        pt_kwargs = pt_kwargs or {}

        from qupulse.pulses import ConstantPT, SequencePT
        current_duration = self.duration
        if callable(to_new_duration):
            new_duration = to_new_duration(current_duration)
        else:
            new_duration = ExpressionScalar(to_new_duration)
        pad_duration = new_duration - current_duration

        if not pt_kwargs and pad_duration == 0:
            return self

        pad_pt = ConstantPT(pad_duration, self.final_values)
        if pt_kwargs or spt_kwargs:
            return SequencePT(self, pad_pt, **pt_kwargs, **spt_kwargs)
        else:
            return self @ pad_pt

    def pad_selected_subtemplates_to(self,
                                     to_new_duration: Union[ExpressionLike, Callable[[Expression], ExpressionLike]],
                                     selector: Callable[['PulseTemplate'], bool] = None,
                                     spt_kwargs: Mapping[str, Any] = None,
                                     identity_map: Callable[[Optional[str]], Optional[str]] = lambda x: x if x is None else f"{x}_padded",
                                   ):
        """Pad all subtemplates for which the selector returns true with the given padding strategy. If no selector is
        specified, all atomic subtemplates are padded. Padding non-atomic pulse templates is generally non-sensical when the subtemplates are padded.

        By default newly created `SequencePT`s have the metadata field `to_single_waveform` set to "always".
        Overwrite `pt_kwargs` to supply other metadata arguments.

        If you need more customization, you can use :py:meth:`.PulseTemplate.with_mapped_subtemplates`.

        !!! Note: This method needs to create new pulse templates and removes all parent tempates identifiers

        Args:
            to_new_duration: Specity how to pad. See :func:`pad_to`.
            selector: Select which subtemplates to pad.
            spt_kwargs: Passed to newly created sequence pulse templates required for padding. By default ``{"metadata": {"to_single_waveform": "always"}}`` is passed to make them atomic.
            identity_map: Provide a function to map the identifiers of composite templates whos subtemplates are padded.

        Returns:
            A new pulse template if any subtemplate needed to be padded.
        """
        if selector is None:
            def selector(pt: PulseTemplate) -> bool:
                return pt._is_atomic()

        if spt_kwargs is None:
            spt_kwargs = {"metadata": {"to_single_waveform": "always"}}

        if selector(self):
            return self.pad_to(to_new_duration, spt_kwargs)

        def map_fn(pt: PulseTemplate) -> PulseTemplate:
            if selector(pt):
                mapped = pt.pad_to(to_new_duration, spt_kwargs)
                return mapped
            else:
                return pt.pad_selected_subtemplates_to(to_new_duration, selector, spt_kwargs)

        return self.with_mapped_subtemplates(map_fn,
                                             identifier_map=identity_map,
                                             recursion_strategy='self')


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
        if self.identifier:
            kwargs = f"identifier={self.identifier!r}," + kwargs
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

    def _get_compare_key(self):
        """Compare pulse templates without runtime meta data."""
        data = self.get_serialization_data()
        data.pop("metadata", None)
        return data

    def __hash__(self):
        if self.__cached_hash_value is None:
            self.__cached_hash_value = forced_hash(self._get_compare_key())
        return self.__cached_hash_value

    def __eq__(self, other):
        if hasattr(other, '_get_compare_key') and hasattr(other, 'metadata'):
            equal = self._get_compare_key() == other._get_compare_key()
            if equal and self.metadata != other.metadata:
                warnings.warn("PulseTemplate differ only in their metadata and are therefore regarded as equal. "
                              "This is required because the metadata field is mutable and the hash implementation requires consistent equality checks.",
                              stacklevel=2,
                              category=MetadataComparison)
            return equal
        else:
            return NotImplemented


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

    def with_parallel_atomic(self, *parallel: 'AtomicPulseTemplate') -> 'AtomicPulseTemplate':
        from qupulse.pulses import AtomicMultiChannelPT
        if parallel:
            return AtomicMultiChannelPT(self, *parallel)
        else:
            return self

    def _is_atomic(self) -> bool:
        return True

    measurement_names = MeasurementDefiner.measurement_names

    def _internal_build_program(self, program_builder: ProgramBuilder):
        context = program_builder.build_context
        scope = context.scope
        assert not scope.get_volatile_parameters().keys() & self.parameter_names, "AtomicPT cannot be volatile"

        waveform = self.build_waveform(parameters=scope,
                                       channel_mapping=context.channel_mapping)
        if not waveform:
            return

        measurements = self.get_measurement_windows(parameters=scope,
                                                    measurement_mapping=context.measurement_mapping)
        program_builder.measure(measurements)
        constant_values = waveform.constant_value_dict()
        if constant_values is None:
            program_builder.play_arbitrary_waveform(waveform)
        else:
            program_builder.hold_voltage(waveform.duration, constant_values)

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


class UnknownVolatileParameter(RuntimeWarning):
    pass


class MetadataComparison(RuntimeWarning):
    pass

class NonStrToSingleWaveformWarning(RuntimeWarning):
    pass
