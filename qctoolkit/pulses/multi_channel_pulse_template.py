"""This module defines MultiChannelPulseTemplate, which allows the combination of several
AtomicPulseTemplates into a single template spanning several channels.

Classes:
    - MultiChannelPulseTemplate: A pulse template defined for several channels by combining pulse
        templates
    - MultiChannelWaveform: A waveform defined for several channels by combining waveforms
"""

from typing import Dict, List, Optional, Any, Iterable, Union, Set, Sequence
import itertools
import numbers

import numpy


from qctoolkit.serialization import Serializer

from qctoolkit.utils.types import MeasurementWindow, ChannelID
from qctoolkit.pulses.instructions import Waveform
from qctoolkit.pulses.pulse_template import PulseTemplate, AtomicPulseTemplate
from qctoolkit.pulses.pulse_template_parameter_mapping import MissingMappingException, MappingPulseTemplate,\
    MissingParameterDeclarationException, MappingTuple
from qctoolkit.pulses.parameters import Parameter, ParameterConstrainer
from qctoolkit.pulses.conditions import Condition
from qctoolkit.expressions import Expression

__all__ = ["MultiChannelWaveform", "AtomicMultiChannelPulseTemplate"]


class MultiChannelWaveform(Waveform):
    """A MultiChannelWaveform is a Waveform object that allows combining arbitrary Waveform objects
    to into a single waveform defined for several channels.

    The number of channels used by the MultiChannelWaveform object is the sum of the channels used
    by the Waveform objects it consists of.

    MultiChannelWaveform allows an arbitrary mapping of channels defined by the Waveforms it
    consists of and the channels it defines. For example, if the MultiChannelWaveform consists
    of a two Waveform objects A and B which define two channels each, then the channels of the
    MultiChannelWaveform may be 0: A.1, 1: B.0, 2: B.1, 3: A.0 where A.0 means channel 0 of Waveform
    object A.

    The following constraints must hold:
     - The durations of all Waveform objects must be equal.
     - The channel mapping must be sane, i.e., no channel of the MultiChannelWaveform must be
        assigned more than one channel of any Waveform object it consists of
    """

    def __init__(self, sub_waveforms: Iterable[Waveform]) -> None:
        """Create a new MultiChannelWaveform instance.

        Requires a list of subwaveforms in the form (Waveform, List(int)) where the list defines
        the channel mapping, i.e., a value y at index x in the list means that channel x of the
        subwaveform will be mapped to channel y of this MultiChannelWaveform object.

        Args:
            sub_waveforms (Iterable( Waveform )): The list of sub waveforms of this
                MultiChannelWaveform
        Raises:
            ValueError, if a channel mapping is out of bounds of the channels defined by this
                MultiChannelWaveform
            ValueError, if several subwaveform channels are assigned to a single channel of this
                MultiChannelWaveform
            ValueError, if subwaveforms have inconsistent durations
        """
        super().__init__()
        if not sub_waveforms:
            raise ValueError(
                "MultiChannelWaveform cannot be constructed without channel waveforms."
            )

        # avoid unnecessary multi channel nesting
        def flatten_sub_waveforms(to_flatten):
            for sub_waveform in to_flatten:
                if isinstance(sub_waveform, MultiChannelWaveform):
                    yield from sub_waveform._sub_waveforms
                else:
                    yield sub_waveform

        # sort the waveforms with their defined channels to make compare key reproducible
        def get_sub_waveform_sort_key(waveform):
            return tuple(sorted(tuple('{}_stringified_numeric_channel'.format(ch) if isinstance(ch, int) else ch
                                      for ch in waveform.defined_channels)))

        self._sub_waveforms = sorted(flatten_sub_waveforms(sub_waveforms),
                                     key=get_sub_waveform_sort_key)

        if not all(waveform.duration == self._sub_waveforms[0].duration for waveform in self._sub_waveforms[1:]):
            raise ValueError(
                "MultiChannelWaveform cannot be constructed from channel waveforms of different"
                "lengths."
            )
        self.__defined_channels = set()
        for waveform in self._sub_waveforms:
            if waveform.defined_channels & self.__defined_channels:
                raise ValueError('Channel may not be defined in multiple waveforms')
            self.__defined_channels |= waveform.defined_channels

    @property
    def duration(self) -> float:
        return self._sub_waveforms[0].duration

    def __getitem__(self, key: ChannelID) -> Waveform:
        for waveform in self._sub_waveforms:
            if key in waveform.defined_channels:
                return waveform
        raise KeyError('Unknown channel ID: {}'.format(key), key)

    @property
    def defined_channels(self) -> Set[ChannelID]:
        return self.__defined_channels

    @property
    def compare_key(self) -> Any:
        # sort with channels
        return tuple(sub_waveform.compare_key for sub_waveform in self._sub_waveforms)

    def unsafe_sample(self,
                      channel: ChannelID,
                      sample_times: numpy.ndarray,
                      output_array: Union[numpy.ndarray, None]=None) -> numpy.ndarray:
        return self[channel].unsafe_sample(channel, sample_times, output_array)

    def get_measurement_windows(self) -> Iterable[MeasurementWindow]:
        return itertools.chain.from_iterable(sub_waveform.get_measurement_windows()
                                             for sub_waveform in self._sub_waveforms)

    def unsafe_get_subset_for_channels(self, channels: Set[ChannelID]) -> 'Waveform':
        relevant_sub_waveforms = tuple(swf for swf in self._sub_waveforms if swf.defined_channels & channels)
        if len(relevant_sub_waveforms) == 1:
            return relevant_sub_waveforms[0].get_subset_for_channels(channels)
        elif len(relevant_sub_waveforms) > 1:
            return MultiChannelWaveform(
                sub_waveform.get_subset_for_channels(channels & sub_waveform.defined_channels)
                for sub_waveform in relevant_sub_waveforms)
        else:
            raise KeyError('Unknown channels: {}'.format(channels))


class AtomicMultiChannelPulseTemplate(AtomicPulseTemplate, ParameterConstrainer):
    def __init__(self,
                 *subtemplates: Union[AtomicPulseTemplate, MappingTuple, MappingPulseTemplate],
                 external_parameters: Optional[Set[str]]=None,
                 identifier: Optional[str]=None,
                 parameter_constraints: Optional[List]=None) -> None:
        AtomicPulseTemplate.__init__(self, identifier=identifier)
        ParameterConstrainer.__init__(self, parameter_constraints=parameter_constraints)

        self._subtemplates = [st if isinstance(st, PulseTemplate) else MappingPulseTemplate.from_tuple(st) for st in
                              subtemplates]

        for subtemplate in self._subtemplates:
            if isinstance(subtemplate, AtomicPulseTemplate):
                continue
            elif isinstance(subtemplate, MappingPulseTemplate):
                if isinstance(subtemplate.template, AtomicPulseTemplate):
                    continue
                else:
                    raise TypeError('Non atomic subtemplate of MappingPulseTemplate: {}'.format(subtemplate.template))
            else:
                raise TypeError('Non atomic subtemplate: {}'.format(subtemplate))

        if not self._subtemplates:
            raise ValueError('Cannot create empty MultiChannelPulseTemplate')

        defined_channels = [st.defined_channels for st in self._subtemplates]

        # check there are no intersections between channels
        for i, channels_i in enumerate(defined_channels):
            for j, channels_j in enumerate(defined_channels[i + 1:]):
                if channels_i & channels_j:
                    raise ChannelMappingException(self._subtemplates[i],
                                                  self._subtemplates[i + 1 + j],
                                                  (channels_i | channels_j).pop())

        if external_parameters is not None:
            remaining = external_parameters.copy()
            for subtemplate in self._subtemplates:
                missing = subtemplate.parameter_names - external_parameters
                if missing:
                    raise MissingParameterDeclarationException(subtemplate, missing.pop())
                remaining -= subtemplate.parameter_names
            missing = self.constrained_parameters - external_parameters
            if missing:
                raise MissingParameterDeclarationException(self, missing.pop())
            remaining -= self.constrained_parameters
            if remaining:
                raise MissingMappingException(self, remaining.pop())

        duration = self._subtemplates[0].duration
        for subtemplate in self._subtemplates[1:]:
            if (duration == subtemplate.duration) is True:
                continue
            else:
                raise ValueError('Could not assert duration equality of {} and {}'.format(duration,
                                                                                          subtemplate.duration))

    @property
    def duration(self) -> Expression:
        return self._subtemplates[0].duration

    @property
    def parameter_names(self) -> Set[str]:
        return set.union(*(st.parameter_names for st in self._subtemplates)) | self.constrained_parameters

    @property
    def subtemplates(self) -> Sequence[AtomicPulseTemplate]:
        return self._subtemplates

    @property
    def defined_channels(self) -> Set[ChannelID]:
        return set.union(*(st.defined_channels for st in self._subtemplates))

    @property
    def measurement_names(self) -> Set[str]:
        return set.union(*(st.measurement_names for st in self._subtemplates))

    def build_waveform(self, parameters: Dict[str, numbers.Real],
                       measurement_mapping: Dict[str, str],
                       channel_mapping: Dict[ChannelID, ChannelID]) -> Optional['MultiChannelWaveform']:
        self.validate_parameter_constraints(parameters=parameters)
        return MultiChannelWaveform(
            [subtemplate.build_waveform(parameters,
                                        measurement_mapping=measurement_mapping,
                                        channel_mapping=channel_mapping) for subtemplate in self._subtemplates])

    def requires_stop(self,
                      parameters: Dict[str, Parameter],
                      conditions: Dict[str, 'Condition']) -> bool:
        return any(st.requires_stop(parameters, conditions) for st in self._subtemplates)

    def get_serialization_data(self, serializer: Serializer) -> Dict[str, Any]:
        data = dict(subtemplates=[serializer.dictify(subtemplate) for subtemplate in self.subtemplates],
                    parameter_constraints=[str(constraint) for constraint in self.parameter_constraints])
        return data

    @staticmethod
    def deserialize(serializer: Serializer,
                    subtemplates: Iterable[Dict[str, Any]],
                    parameter_constraints: Any,
                    identifier: Optional[str] = None) -> 'AtomicMultiChannelPulseTemplate':
        subtemplates = [serializer.deserialize(st) for st in subtemplates]
        return AtomicMultiChannelPulseTemplate(*subtemplates,
                                               parameter_constraints=parameter_constraints,
                                               identifier=identifier)


class ChannelMappingException(Exception):
    def __init__(self, obj1, obj2, intersect_set):
        self.intersect_set = intersect_set
        self.obj1 = obj1
        self.obj2 = obj2

    def __str__(self) -> str:
        return 'Channels {chs} defined in {o1} and {o2}'.format(chs=self.intersect_set, o1=self.obj1, o2=self.obj2)