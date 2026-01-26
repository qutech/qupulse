"""Definition of the program builder protocol."""
import copy
import dataclasses
from abc import abstractmethod, ABC
from contextlib import contextmanager
from typing import runtime_checkable, Protocol, Mapping, Optional, Sequence, Iterable, ContextManager, AbstractSet, \
    Union

from qupulse import MeasurementWindow
from qupulse.expressions import Expression
from qupulse.parameter_scope import Scope, MappedScope
from qupulse.program.waveforms import Waveform, ConstantWaveform, TransformingWaveform
from qupulse.program.transformation import Transformation, chain_transformations
from qupulse.program.values import RepetitionCount, HardwareTime, HardwareVoltage
from qupulse.pulses.metadata import TemplateMetadata

from qupulse.utils.types import TimeType, ChannelID


@runtime_checkable
class Program(Protocol):
    """This protocol is used to inspect and or manipulate programs. As you can see the functionality is very limited
    because most of a program class' capability are specific to the implementation."""

    @property
    @abstractmethod
    def duration(self) -> TimeType:
        """The duration of the program in nanoseconds."""

    @abstractmethod
    def get_defined_channels(self) -> AbstractSet[ChannelID]:
        """Get the set of channels that are used in this program."""


@dataclasses.dataclass
class BuildContext:
    scope: Scope = None
    measurement_mapping: Mapping[str, Optional[str]] = None
    channel_mapping: Mapping[ChannelID, Optional[ChannelID]] = None
    transformation: Optional[Transformation] = None
    minimal_sample_rate: Optional[TimeType] = None

    def apply_mappings(self,
                       parameter_mapping: Mapping[str, Expression] = None,
                       measurement_mapping: Mapping[str, Optional[str]] = None,
                       channel_mapping: Mapping[ChannelID, Optional[ChannelID]] = None,
                       ) -> "BuildContext":
        scope = self.scope
        if parameter_mapping is not None:
            scope = MappedScope(scope=scope, mapping=parameter_mapping)
        mapped_measurement_mapping = self.measurement_mapping
        if measurement_mapping is not None:
            # bruh
            mapped_measurement_mapping = {k: mapped_measurement_mapping[v] for k, v in measurement_mapping.items()}
        mapped_channel_mapping = self.channel_mapping
        if channel_mapping is not None:
            mapped_channel_mapping = {inner_ch: None if outer_ch is None else mapped_channel_mapping[outer_ch]
                                      for inner_ch, outer_ch in channel_mapping.items()}
        return BuildContext(scope=scope, measurement_mapping=mapped_measurement_mapping, channel_mapping=mapped_channel_mapping, transformation=self.transformation, minimal_sample_rate=self.minimal_sample_rate)


@dataclasses.dataclass
class BuildSettings:
    to_single_waveform: AbstractSet[str | object]


@runtime_checkable
class ProgramBuilder(Protocol):
    """This protocol is used by :py:meth:`.PulseTemplate.create_program` to build a program via a variation of the
    visitor pattern.

    The pulse templates call the methods that correspond to their functionality on the program builder. For example,
    :py:class:`.ConstantPulseTemplate` translates itself into a simple :py:meth:`.ProgramBuilder.hold_voltage` call while
    :class:`SequencePulseTemplate` uses :py:meth:`.ProgramBuilder.with_sequence` to signify a logical unit with
    attached measurements and passes the resulting object to the sequenced sub-templates.

    Due to backward compatibility the handling of measurements is a bit weird since they have to be omitted in certain
    cases. However, this is not relevant for HDAWG specific implementations because these are expected to ignore
    :py:meth:`.ProgramBuilder.measure` calls.

    This interface makes heavy use of context managers and generators/iterators which allows for flexible iteration
    and repetition implementation.
    """

    @property
    @abstractmethod
    def build_context(self) -> BuildContext:
        """Get the current build context."""

    @property
    @abstractmethod
    def build_settings(self) -> BuildSettings:
        """Get the current build settings"""

    @abstractmethod
    def override(self,
                 scope: Scope = None,
                 measurement_mapping: Optional[Mapping[str, Optional[str]]] = None,
                 channel_mapping: Optional[Mapping[ChannelID, Optional[ChannelID]]] = None,
                 global_transformation: Optional[Transformation] = None,
                 to_single_waveform: AbstractSet[str | object] = None,):
        """Override the non-None values in context and settings"""


    @abstractmethod
    def with_mappings(self, *,
                      parameter_mapping: Mapping[str, Expression],
                      measurement_mapping: Mapping[str, Optional[str]],
                      channel_mapping: Mapping[ChannelID, Optional[ChannelID]],
                      ) -> ContextManager['ProgramBuilder']:
        """Modify the build context for the duration of the context manager.

        Args:
            parameter_mapping: A mapping of parameter names to expressions.
            measurement_mapping: A mapping of measurement names to measurement names or None.
            channel_mapping: A mapping of channel IDs to channel IDs or None.
        """

    @abstractmethod
    def with_transformation(self, transformation: Transformation) -> ContextManager['ProgramBuilder']:
        """Modify the build context for the duration of the context manager."""

    @abstractmethod
    def with_metadata(self, metadata: TemplateMetadata) -> ContextManager['ProgramBuilder']:
        """Modify the build context for the duration of the context manager."""

    @abstractmethod
    def hold_voltage(self, duration: HardwareTime, voltages: Mapping[str, HardwareVoltage]):
        """Hold the specified voltage for a given time. Advances the current time by ``duration``. The values are
        hardware dependent type which are inserted into the parameter scope via :py:meth:`.ProgramBuilder.with_iteration`.

        Args:
            duration: Duration of voltage hold
            voltages: Voltages for each channel
        """

    # further specialized commandos like play_harmonic might be added here

    @abstractmethod
    def play_arbitrary_waveform(self, waveform: Waveform):
        """Insert the playback of an arbitrary waveform. If possible pulse templates should use more specific commands
        like :py:meth:`.ProgramBuilder.hold_voltage` (the only more specific command at the time of this writing).

        Args:
            waveform: The waveform to play
        """

    @abstractmethod
    def measure(self, measurements: Optional[Sequence[MeasurementWindow]]):
        """Unconditionally add given measurements relative to the current position.

        Args:
            measurements: Measurements to add.
        """

    @abstractmethod
    def with_repetition(self, repetition_count: RepetitionCount,
                        measurements: Optional[Sequence[MeasurementWindow]] = None) -> Iterable['ProgramBuilder']:
        """Start a new repetition context with given repetition count. The caller has to iterate over the return value
        and call `:py:meth:`.ProgramBuilder.inner_scope` inside the iteration context.

        Args:
            repetition_count: Repetition count
            measurements: These measurements are added relative to the position at the start of the iteration iff the
                          iteration is not empty.

        Returns:
            An iterable of :py:class:`ProgramBuilder` instances.
        """

    @abstractmethod
    def with_sequence(self,
                      measurements: Optional[Sequence[MeasurementWindow]] = None) -> ContextManager['ProgramBuilder']:
        """Start a new sequence context. The caller has to enter the returned context manager and add the sequenced
        elements there.

        Measurements that are added in to the returned program builder are discarded if the sequence is empty on exit.

        Args:
            measurements: These measurements are added relative to the position at the start of the sequence iff the
            sequence is not empty.

        Returns:
            A context manager that returns a :py:class:`ProgramBuilder` on entering.
        """

    @abstractmethod
    def new_subprogram(self) -> ContextManager['ProgramBuilder']:
        """Create a context managed program builder whose contents are translated into a single waveform upon exit if
        it is not empty.

        Returns:
            A context manager that returns a :py:class:`ProgramBuilder` on entering.
        """

    @abstractmethod
    def with_iteration(self, index_name: str, rng: range,
                       measurements: Optional[Sequence[MeasurementWindow]] = None) -> Iterable['ProgramBuilder']:
        """Create an iterable that represent the body of the iteration. This can be an iterable with an element for each
        step in the iteration or a single object that represents the complete iteration.

        Args:
            index_name: The name of index
            rng: The range if the index
            measurements: Measurements to add iff the iteration body is not empty.
        """

    @abstractmethod
    def time_reversed(self) -> ContextManager['ProgramBuilder']:
        """This returns a new context manager that will reverse everything added to it in time upon exit.

        Returns:
            A context manager that returns a :py:class:`ProgramBuilder` on entering.
        """

    @abstractmethod
    def to_program(self) -> Optional[Program]:
        """Generate the final program. This is allowed to invalidate the program builder.

        Returns:
            A program implementation. None if nothing was added to this program builder.
        """


class BaseProgramBuilder(ProgramBuilder, ABC):
    def __init__(self, initial_context: BuildContext = None, initial_settings: BuildSettings = None):
        self._build_context_stack: list[BuildContext] = [BuildContext() if initial_context is None else initial_context]
        self._build_settings_stack: list[BuildSettings] = [BuildSettings(set()) if initial_settings is None else initial_settings]

    @property
    def build_context(self) -> BuildContext:
        return self._build_context_stack[-1]

    @property
    def build_settings(self) -> BuildSettings:
        return self._build_settings_stack[-1]

    def override(self,
                 scope: Scope = None,
                 measurement_mapping: Optional[Mapping[str, Optional[str]]] = None,
                 channel_mapping: Optional[Mapping[ChannelID, Optional[ChannelID]]] = None,
                 global_transformation: Optional[Transformation] = None,
                 to_single_waveform: AbstractSet[Union[str, 'PulseTemplate']] = None):
        old_context = self._build_context_stack[-1]
        context = BuildContext(
            scope=old_context.scope if scope is None else scope,
            measurement_mapping=old_context.measurement_mapping if measurement_mapping is None else measurement_mapping,
            channel_mapping=old_context.channel_mapping if channel_mapping is None else channel_mapping,
            transformation=old_context.transformation if global_transformation is None else global_transformation,
        )
        old_settings = self._build_settings_stack[-1]
        settings = BuildSettings(
            to_single_waveform=old_settings.to_single_waveform if to_single_waveform is None else to_single_waveform,
        )

        self._build_context_stack.append(context)
        self._build_settings_stack.append(settings)

    @contextmanager
    def _with_patched_context(self, **kwargs):
        context = copy.copy(self._build_context_stack[-1])
        for name, value in kwargs.items():
            setattr(context, name, value)
        self._build_context_stack.append(context)
        yield
        self._build_context_stack.pop()

    @contextmanager
    def with_metadata(self, metadata: TemplateMetadata):
        """to single waveform

        can be set in the pulse_template metadata or in the build_settings

         - Should the program_builder know which template is translated currently?
        """

        # metadata.to_single_waveform == "always" is handled in PulseTemplate._build_program
        if metadata.minimal_sample_rate is not None:
            with self._with_patched_context(minimal_sample_rate=metadata.minimal_sample_rate) as builder:
                yield builder
        else:
            yield self

    @contextmanager
    def with_transformation(self, transformation: Transformation):
        context = copy.copy(self.build_context)
        context.transformation = chain_transformations(context.transformation, transformation)
        self._build_context_stack.append(context)
        yield self
        self._build_context_stack.pop()

    @contextmanager
    def with_mappings(self, *,
                      parameter_mapping: Mapping[str, Expression],
                      measurement_mapping: Mapping[str, Optional[str]],
                      channel_mapping: Mapping[ChannelID, Optional[ChannelID]],
                      ):
        context = self.build_context.apply_mappings(parameter_mapping, measurement_mapping, channel_mapping)
        self._build_context_stack.append(context)
        yield self
        self._build_context_stack.pop()

    @abstractmethod
    def _transformed_hold_voltage(self, duration: HardwareTime, voltages: Mapping[str, HardwareVoltage]):
        """"""

    @abstractmethod
    def _transformed_play_arbitrary_waveform(self, waveform: Waveform):
        """"""

    def play_arbitrary_waveform(self, waveform: Waveform):
        transformation = self.build_context.transformation
        if transformation:
            transformed_waveform = TransformingWaveform(waveform, transformation)
            self._transformed_play_arbitrary_waveform(transformed_waveform)
        else:
            self._transformed_play_arbitrary_waveform(waveform)

    def hold_voltage(self, duration: HardwareTime, voltages: Mapping[str, HardwareVoltage]):
        transformation = self.build_context.transformation
        if transformation:
            if transformation.get_constant_output_channels(voltages.keys()) != transformation.get_output_channels(voltages.keys()):
                waveform = TransformingWaveform(ConstantWaveform.from_mapping(duration, voltages), transformation)
                self._transformed_play_arbitrary_waveform(waveform)
            else:
                transformed_voltages = transformation(0.0, voltages)
                self._transformed_hold_voltage(duration, transformed_voltages)
        else:
            self._transformed_hold_voltage(duration, voltages)
