"""This module defines SequencePulseTemplate, a higher-order hierarchical pulse template that
combines several other PulseTemplate objects for sequential execution."""

import numpy as np
from typing import Dict, List, Set, Optional, Any, Iterable, Union, Callable, cast
from numbers import Real
import functools
import warnings

from qupulse.serialization import Serializer, PulseRegistryType
from qupulse._program._loop import Loop
from qupulse.parameter_scope import Scope
from qupulse.utils import cached_property
from qupulse.utils.types import MeasurementWindow, ChannelID, TimeType
from qupulse.pulses.pulse_template import PulseTemplate, AtomicPulseTemplate
from qupulse.pulses.parameters import Parameter, ParameterConstrainer, ParameterNotProvidedException
from qupulse.pulses.mapping_pulse_template import MappingPulseTemplate, MappingTuple
from qupulse._program.waveforms import SequenceWaveform
from qupulse.pulses.measurement import MeasurementDeclaration, MeasurementDefiner
from qupulse.expressions import Expression, ExpressionScalar

__all__ = ["SequencePulseTemplate"]


class SequencePulseTemplate(PulseTemplate, ParameterConstrainer, MeasurementDefiner):
    """A sequence of different PulseTemplates.
    
    SequencePulseTemplate allows to group several
    PulseTemplates (subtemplates) into one larger sequence,
    i.e., when instantiating a pulse from a SequencePulseTemplate
    all pulses instantiated from the subtemplates are queued for
    execution right after one another.
    SequencePulseTemplate requires to specify a mapping of
    parameter declarations from its subtemplates to its own, enabling
    renaming and mathematical transformation of parameters.
    """

    def __init__(self,
                 *subtemplates: Union[PulseTemplate, MappingTuple],
                 identifier: Optional[str]=None,
                 parameter_constraints: Optional[List[Union[str, Expression]]]=None,
                 measurements: Optional[List[MeasurementDeclaration]]=None,
                 registry: PulseRegistryType=None) -> None:
        """Create a new SequencePulseTemplate instance.

        Requires a (correctly ordered) list of subtemplates in the form
        (PulseTemplate, Dict(str -> str)) where the dictionary is a mapping between the external
        parameters exposed by this SequencePulseTemplate to the parameters declared by the
        subtemplates, specifying how the latter are derived from the former, i.e., the mapping is
        subtemplate_parameter_name -> mapping_expression (as str) where the free variables in the
        mapping_expression are parameters declared by this SequencePulseTemplate.

        The following requirements must be satisfied:
            - for each parameter declared by a subtemplate, a mapping expression must be provided
            - each free variable in a mapping expression must be declared as an external parameter
                of this SequencePulseTemplate

        Args:
            subtemplates (List(Subtemplate)): The list of subtemplates of this
                SequencePulseTemplate as tuples of the form (PulseTemplate, Dict(str -> str)).
            identifier (str): A unique identifier for use in serialization. (optional)
        """
        PulseTemplate.__init__(self, identifier=identifier)
        ParameterConstrainer.__init__(self, parameter_constraints=parameter_constraints)
        MeasurementDefiner.__init__(self, measurements=measurements)

        self.__subtemplates = [MappingPulseTemplate.from_tuple(st) if isinstance(st, tuple) else st
                               for st in subtemplates]

        # check that all subtemplates live on the same channels
        defined_channels = self.__subtemplates[0].defined_channels
        for subtemplate in self.__subtemplates[1:]:
            if subtemplate.defined_channels != defined_channels:
                raise ValueError('The subtemplates are defined for different channels:'
                                 + f' defined {defined_channels} vs. subtemplate {subtemplate.defined_channels}')

        self._register(registry=registry)

    @classmethod
    def concatenate(cls, *pulse_templates: Union[PulseTemplate, MappingTuple], **kwargs) -> 'SequencePulseTemplate':
        """Sequences the given pulse templates by creating a SequencePulseTemplate. Pulse templates that are
        SequencePulseTemplates and do not carry additional information (identifier, measurements, parameter constraints)
        are not used directly but their sub templates are.
        Args:
            *pulse_templates: Pulse templates to concatenate
            **kwargs: Forwarded to the __init__

        Returns: Concatenated templates
        """
        parsed = []
        for pt in pulse_templates:
            if (isinstance(pt, SequencePulseTemplate)
                    and pt.identifier is None
                    and not pt.measurement_declarations
                    and not pt.parameter_constraints):
                parsed.extend(pt.subtemplates)
            else:
                parsed.append(pt)
        return cls(*parsed, **kwargs)

    @property
    def parameter_names(self) -> Set[str]:
        return self.constrained_parameters.union(self.measurement_parameters, *(st.parameter_names for st in self.__subtemplates))

    @property
    def subtemplates(self) -> List[MappingPulseTemplate]:
        return self.__subtemplates

    @cached_property
    def duration(self) -> Expression:
        return sum(sub.duration for sub in self.__subtemplates)

    @property
    def defined_channels(self) -> Set[ChannelID]:
        return self.__subtemplates[0].defined_channels if self.__subtemplates else set()

    @property
    def measurement_names(self) -> Set[str]:
        return set.union(MeasurementDefiner.measurement_names.fget(self),
                         *(st.measurement_names for st in self.subtemplates))

    def build_waveform(self,
                       parameters: Dict[str, Real],
                       channel_mapping: Dict[ChannelID, ChannelID]) -> SequenceWaveform:
        self.validate_parameter_constraints(parameters=parameters, volatile=set())
        return SequenceWaveform.from_sequence(
            [sub_template.build_waveform(parameters, channel_mapping=channel_mapping)
             for sub_template in self.__subtemplates])

    def _internal_create_program(self, *,
                                 scope: Scope,
                                 measurement_mapping: Dict[str, Optional[str]],
                                 channel_mapping: Dict[ChannelID, Optional[ChannelID]],
                                 global_transformation: Optional['Transformation'],
                                 to_single_waveform: Set[Union[str, 'PulseTemplate']],
                                 parent_loop: Loop) -> None:
        self.validate_scope(scope)

        if self.duration.evaluate_in_scope(scope) > 0:
            measurements = self.get_measurement_windows(scope, measurement_mapping)
            if measurements:
                parent_loop.add_measurements(measurements)

            for subtemplate in self.subtemplates:
                subtemplate._create_program(scope=scope,
                                            measurement_mapping=measurement_mapping,
                                            channel_mapping=channel_mapping,
                                            global_transformation=global_transformation,
                                            to_single_waveform=to_single_waveform,
                                            parent_loop=parent_loop)

    def get_serialization_data(self, serializer: Optional[Serializer]=None) -> Dict[str, Any]:
        data = super().get_serialization_data(serializer)
        data['subtemplates'] = self.subtemplates

        if serializer: # compatibility to old serialization routines, deprecated
            data = dict()
            data['subtemplates'] = [serializer.dictify(subtemplate) for subtemplate in self.subtemplates]

        if self.parameter_constraints:
            data['parameter_constraints'] = [str(c) for c in self.parameter_constraints]
        if self.measurement_declarations:
            data['measurements'] = self.measurement_declarations

        return data

    @classmethod
    def deserialize(cls,
                    serializer: Optional[Serializer]=None,  # compatibility to old serialization routines, deprecated
                    **kwargs) -> 'SequencePulseTemplate':
        subtemplates = kwargs['subtemplates']
        del kwargs['subtemplates']

        if serializer: # compatibility to old serialization routines, deprecated
            subtemplates = [serializer.deserialize(st) for st in subtemplates]

        return cls(*subtemplates, **kwargs)

    @property
    def defined_channels(self) -> Set[ChannelID]:
        return self.__subtemplates[0].defined_channels

    @property
    def integral(self) -> Dict[ChannelID, ExpressionScalar]:
        expressions = {channel: 0 for channel in self.defined_channels}

        def add_dicts(x, y):
            return {k: x[k] + y[k] for k in x}

        return functools.reduce(add_dicts, [sub.integral for sub in self.__subtemplates], expressions)

    @property
    def initial_values(self) -> Dict[ChannelID, ExpressionScalar]:
        return self.__subtemplates[0].initial_values

    @property
    def final_values(self) -> Dict[ChannelID, ExpressionScalar]:
        return self.__subtemplates[-1].final_values

