"""This module defines SequencePulseTemplate, a higher-order hierarchical pulse template that
combines several other PulseTemplate objects for sequential execution."""

import numpy as np
from typing import Dict, List, Tuple, Set, Optional, Any, Iterable, Union
from numbers import Real
import functools
import warnings

from cached_property import cached_property

from qupulse.serialization import Serializer, PulseRegistryType
from qupulse._program._loop import Loop

from qupulse.utils.types import MeasurementWindow, ChannelID, TimeType
from qupulse.pulses.pulse_template import PulseTemplate
from qupulse.pulses.parameters import Parameter, ParameterConstrainer, ParameterNotProvidedException
from qupulse.pulses.sequencing import InstructionBlock, Sequencer
from qupulse.pulses.conditions import Condition
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
                 external_parameters: Optional[Union[Iterable[str], Set[str]]]=None,
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
            external_parameters (List(str)): A set of names for external parameters of this
                SequencePulseTemplate. Deprecated.
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
                raise ValueError('The subtemplates are defined for different channels')

        if external_parameters:
            warnings.warn("external_parameters is an obsolete argument and will be removed in the future.",
                          category=DeprecationWarning)

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

    @property
    def is_interruptable(self) -> bool:
        return any(st.is_interruptable for st in self.subtemplates)

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

    def requires_stop(self,
                      parameters: Dict[str, Parameter],
                      conditions: Dict[str, 'Condition']) -> bool:
        """Returns the stop requirement of the first subtemplate. If a later subtemplate requires a stop the
        SequencePulseTemplate can be partially sequenced."""
        return self.__subtemplates[0].requires_stop(parameters, conditions) if self.__subtemplates else False

    def build_waveform(self,
                       parameters: Dict[str, Real],
                       channel_mapping: Dict[ChannelID, ChannelID]) -> SequenceWaveform:
        self.validate_parameter_constraints(parameters=parameters)
        return SequenceWaveform([sub_template.build_waveform(parameters,
                                                             channel_mapping=channel_mapping)
                                 for sub_template in self.__subtemplates])

    def build_sequence(self,
                       sequencer: Sequencer,
                       parameters: Dict[str, Parameter],
                       conditions: Dict[str, Condition],
                       measurement_mapping: Dict[str, str],
                       channel_mapping: Dict['ChannelID', 'ChannelID'],
                       instruction_block: InstructionBlock) -> None:
        self.validate_parameter_constraints(parameters=parameters)
        self.insert_measurement_instruction(instruction_block=instruction_block,
                                            parameters=parameters,
                                            measurement_mapping=measurement_mapping)
        for subtemplate in reversed(self.subtemplates):
            sequencer.push(subtemplate,
                           parameters=parameters,
                           conditions=conditions,
                           window_mapping=measurement_mapping,
                           channel_mapping=channel_mapping,
                           target_block=instruction_block)

    def _internal_create_program(self, *,
                                 parameters: Dict[str, Parameter],
                                 measurement_mapping: Dict[str, Optional[str]],
                                 channel_mapping: Dict[ChannelID, Optional[ChannelID]],
                                 global_transformation: Optional['Transformation'],
                                 to_single_waveform: Set[Union[str, 'PulseTemplate']],
                                 parent_loop: Loop) -> None:
        self.validate_parameter_constraints(parameters=parameters)

        try:
            measurement_parameters = {parameter_name: parameters[parameter_name].get_value()
                                      for parameter_name in self.measurement_parameters}
            duration_parameters = {parameter_name: parameters[parameter_name].get_value()
                                   for parameter_name in self.duration.variables}
        except KeyError as e:
            raise ParameterNotProvidedException(e) from e

        if self.duration.evaluate_numeric(**duration_parameters) > 0:
            measurements = self.get_measurement_windows(measurement_parameters, measurement_mapping)
            if measurements:
                parent_loop.add_measurements(measurements)

            for subtemplate in self.subtemplates:
                subtemplate._create_program(parameters=parameters,
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


