from typing import Set, Optional, Dict

from qupulse import ChannelID
from qupulse.expressions import ExpressionScalar
from qupulse.serialization import PulseRegistryType
from qupulse.pulses.pulse_template import PulseTemplate


class AbstractPulseTemplate(PulseTemplate):
    def __init__(self, *,
                 identifier: str,
                 defined_channels: Set[ChannelID],

                 parameter_names: Optional[Set[str]]=None,
                 measurement_names: Optional[Set[str]]=None,
                 integral: Optional[Dict[ChannelID, ExpressionScalar]]=None,
                 duration: Optional[ExpressionScalar]=None,
                 is_interruptable: Optional[bool]=None,

                 registry: Optional[PulseRegistryType]=None):
        """
        Guarantee:
        A property whose get method was called always returns the same value

        Mandatory properties:
          - identifier
          - defined_channels

        Args:
            identifier:
            defined_channels:
            measurement_names:
            integral:
            duration:
        """
        super().__init__(identifier=identifier)

        # fixed property
        self._declared_properties = {'defined_channels': set(defined_channels)}
        self._frozen_properties = {'defined_channels'}

        if parameter_names is not None:
            self._declared_properties['parameter_names'] = set(map(str, parameter_names))

        if measurement_names is not None:
            self._declared_properties['measurement_names'] = set(map(str, measurement_names))

        if integral is not None:
            if integral.keys() != defined_channels:
                raise ValueError('Integral does not fit to defined channels', integral.keys(), defined_channels)
            self._declared_properties['integral'] = {channel: ExpressionScalar(value)
                                                     for channel, value in integral.items()}

        if duration:
            self._declared_properties['duration'] = ExpressionScalar(duration)

        if is_interruptable is not None:
            self._declared_properties['is_interruptable'] = bool(is_interruptable)

        self._linked_target = None

        self._register(registry=registry)

    def link_to(self, target: PulseTemplate):
        if self._linked_target:
            raise RuntimeError('Cannot is already linked. Cannot relink once linked AbstractPulseTemplate.')

        for frozen_property in self._frozen_properties:
            if self._declared_properties[frozen_property] != getattr(target, frozen_property):
                raise RuntimeError('Cannot link to target. Wrong value of property "%s"' % frozen_property)

        self._linked_target = target

    def get_serialization_data(self, serializer=None) -> Dict:
        if serializer:
            raise RuntimeError('Old serialization not supported in new class')

        data = super().get_serialization_data()
        data.update(self._declared_properties)
        return data

    @staticmethod
    def _freezing_property(property_name):
        @property
        def property_getter(self: 'AbstractPulseTemplate'):
            if self._linked_target:
                return getattr(self._linked_target, property_name)
            elif property_name in self._declared_properties:
                self._frozen_properties.add(property_name)
                return self._declared_properties[property_name]
            else:
                raise NotSpecifiedError(property_name)
        return property_getter

    def _internal_create_program(self, **kwargs):
        raise NotImplementedError('this should never be called as we overrode _create_program')

    def _create_program(self, **kwargs):
        if self._linked_target:
            return self._linked_target._create_program(**kwargs)
        else:
            raise RuntimeError('No linked target to refer to')

    defined_channels = _freezing_property('defined_channels')
    duration = _freezing_property('duration')
    measurement_names = _freezing_property('measurement_names')
    integral = _freezing_property('integral')
    parameter_names = _freezing_property('parameter_names')


class NotSpecifiedError(RuntimeError):
    pass
