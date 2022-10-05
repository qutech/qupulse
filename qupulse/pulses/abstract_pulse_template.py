from typing import Set, Optional, Dict, Any, cast
from functools import partial, partialmethod
import warnings

from qupulse import ChannelID
from qupulse.expressions import ExpressionScalar
from qupulse.serialization import PulseRegistryType, Serializable
from qupulse.pulses.pulse_template import PulseTemplate


__all__ = ["AbstractPulseTemplate", "UnlinkWarning"]


class AbstractPulseTemplate(PulseTemplate):
    _PROPERTY_DOC = """Abstraction of :py:attr:`.PulseTemplate.{name}`. Raises :class:`.NotSpecifiedError` if the
    abstract template is unlinked or the property was not specified."""

    def __init__(self, identifier: str,
                 *,
                 defined_channels: Optional[Set[ChannelID]]=None,
                 parameter_names: Optional[Set[str]]=None,
                 measurement_names: Optional[Set[str]]=None,
                 integral: Optional[Dict[ChannelID, ExpressionScalar]]=None,
                 duration: Optional[ExpressionScalar]=None,
                 registry: Optional[PulseRegistryType]=None):
        """This pulse template can be used as a place holder for a pulse template with a defined interface. Pulse
        template properties like :func:`defined_channels` can be passed on initialization to declare those properties who make
        up the interface. Omitted properties raise an :class:`.NotSpecifiedError` exception if accessed. Properties
        which have been accessed are marked as "frozen".

        The abstract pulse template can be linked to another pulse template by calling the `link_to` member. The target
        has to have the same properties for all properties marked as "frozen". This ensures a property always returns
        the same value.

        Example:
            >>> abstract_readout = AbstractPulseTemplate('readout', defined_channels={'X', 'Y'})
            >>> assert abstract_readout.defined_channels == {'X', 'Y'}
            >>> # This will raise an exception because duration is not specified
            >>> print(abstract_readout.duration)

        Args:
            identifier: Mandatory property
            defined_channels: Optional property
            parameter_names: Optional property
            measurement_names: Optional property
            integral: Optional property
            duration: Optional property
            registry: Instance is registered here if specified
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

        self._linked_target = None
        self.serialize_linked = False

        self._register(registry=registry)

    def link_to(self, target: PulseTemplate, serialize_linked: bool=None):
        """Link to another pulse template.

        Args:
            target: Forward all getattr calls to this pulse template
            serialize_linked: If true, serialization will be forwarded. Otherwise serialization will ignore the link
        """
        if self._linked_target:
            raise RuntimeError('Cannot is already linked. If you REALLY need to relink call unlink() first.')

        for frozen_property in self._frozen_properties:
            if self._declared_properties[frozen_property] != getattr(target, frozen_property):
                raise RuntimeError('Cannot link to target. Wrong value of property "%s"' % frozen_property)

        if serialize_linked is not None:
            self.serialize_linked = serialize_linked
        self._linked_target = target

    def unlink(self):
        """Unlink a linked target. This might lead to unexpected behaviour as forwarded get attributes are not frozen"""
        if self._linked_target:
            warnings.warn("This might lead to unexpected behaviour as forwarded attributes are not frozen. Parent pulse"
                          " templates might rely on certain properties to be constant (for example due to caching).",
                          UnlinkWarning)
        self._linked_target = None

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
            raise NotSpecifiedError(self.identifier, property_name)

    def _forward_if_linked(self, method_name: str, *args, **kwargs) -> Any:
        if self._linked_target:
            return getattr(self._linked_target, method_name)(*args, **kwargs)
        else:
            raise RuntimeError('Cannot call "%s". No linked target to refer to', method_name)

    def _internal_create_program(self, **kwargs):
        raise NotImplementedError('this should never be called as we overrode _create_program')  # pragma: no cover

    _create_program = partialmethod(_forward_if_linked, '_create_program')

    defined_channels = property(partial(_get_property, property_name='defined_channels'),
                                doc=_PROPERTY_DOC.format(name='defined_channels'))
    duration = property(partial(_get_property, property_name='duration'),
                        doc=_PROPERTY_DOC.format(name='duration'))
    measurement_names = property(partial(_get_property, property_name='measurement_names'),
                                 doc=_PROPERTY_DOC.format(name='measurement_names'))
    integral = property(partial(_get_property, property_name='integral'),
                        doc=_PROPERTY_DOC.format(name='integral'))
    parameter_names = property(partial(_get_property, property_name='parameter_names'),
                               doc=_PROPERTY_DOC.format(name='parameter_names'))
    initial_values = property(partial(_get_property, property_name='initial_values'),
                              doc=_PROPERTY_DOC.format(name='initial_values'))
    final_values = property(partial(_get_property, property_name='final_values'),
                            doc=_PROPERTY_DOC.format(name='final_values'))

    __hash__ = None


class NotSpecifiedError(RuntimeError):
    pass


class UnlinkWarning(UserWarning):
    pass
