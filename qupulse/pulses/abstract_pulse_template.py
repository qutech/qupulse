from typing import Set, Optional, Dict, Any, cast
from functools import partial, partialmethod

from qupulse import ChannelID
from qupulse.expressions import ExpressionScalar
from qupulse.serialization import PulseRegistryType
from qupulse.pulses.pulse_template import PulseTemplate


class AbstractPulseTemplate(PulseTemplate):
    def __init__(self, *,
                 identifier: str,

                 defined_channels: Set[ChannelID]=None,
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

        Args:
            defined_channels
            identifier:
            defined_channels:
            measurement_names:
            integral:
            duration:
        """
        super().__init__(identifier=identifier)

        self._declared_properties = {}
        self._frozen_properties = set()

        if defined_channels is not None:
            self._declared_properties['defined_channels'] = set(defined_channels)

        if parameter_names is not None:
            self._declared_properties['parameter_names'] = set(map(str, parameter_names))

        if measurement_names is not None:
            self._declared_properties['measurement_names'] = set(map(str, measurement_names))

        if integral is not None:
            if defined_channels is not None and integral.keys() != defined_channels:
                raise ValueError('Integral does not fit to defined channels', integral.keys(), defined_channels)
            self._declared_properties['integral'] = {channel: ExpressionScalar(value)
                                                     for channel, value in integral.items()}

        if duration:
            self._declared_properties['duration'] = ExpressionScalar(duration)

        if is_interruptable is not None:
            self._declared_properties['is_interruptable'] = bool(is_interruptable)

        self._linked_target = None
        self.serialize_linked = False

        self._register(registry=registry)

    def link_to(self, target: PulseTemplate, serialize_linked: bool=None):
        if self._linked_target:
            raise RuntimeError('Cannot is already linked. Cannot relink once linked AbstractPulseTemplate.')

        for frozen_property in self._frozen_properties:
            if self._declared_properties[frozen_property] != getattr(target, frozen_property):
                raise RuntimeError('Cannot link to target. Wrong value of property "%s"' % frozen_property)

        if serialize_linked is not None:
            self.serialize_linked = serialize_linked
        self._linked_target = target

    def __getattr__(self, item: str) -> Any:
        """Forward all unknown attribute accesses."""
        return getattr(self._linked_target, item)

    def get_serialization_data(self, serializer=None) -> Dict:
        if self._linked_target and self.serialize_linked:
            return self._linked_target.get_serialization_data(serializer=serializer)

        if serializer:
            raise RuntimeError('Old serialization not supported in new class')

        data = super().get_serialization_data()
        data.update(self._declared_properties)
        return data

    def _get_property(self, property_name: str) -> Any:
        if self._linked_target:
            return getattr(self._linked_target, property_name)
        elif property_name in self._declared_properties:
            self._frozen_properties.add(property_name)
            return self._declared_properties[property_name]
        else:
            raise NotSpecifiedError(property_name)

    def _forward_if_linked(self, method_name: str, *args, **kwargs) -> Any:
        if self._linked_target:
            return getattr(self._linked_target, method_name)(*args, **kwargs)
        else:
            raise RuntimeError('Cannot call "%s". No linked target to refer to', method_name)

    def _internal_create_program(self, **kwargs):
        raise NotImplementedError('this should never be called as we overrode _create_program')

    _create_program = partialmethod(_forward_if_linked, '_create_program')
    build_sequence = partialmethod(_forward_if_linked, 'build_sequence')
    is_interruptable = partialmethod(_forward_if_linked, 'is_interruptable')
    requires_stop = partialmethod(_forward_if_linked, 'requires_stop')

    defined_channels = property(partial(_get_property, property_name='defined_channels'))
    duration = property(partial(_get_property, property_name='duration'))
    measurement_names = property(partial(_get_property, property_name='measurement_names'))
    integral = property(partial(_get_property, property_name='integral'))
    parameter_names = property(partial(_get_property, property_name='parameter_names'))


class NotSpecifiedError(RuntimeError):
    pass
