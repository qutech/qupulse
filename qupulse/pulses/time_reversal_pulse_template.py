from typing import Optional, Set, Dict, Union

from qupulse import ChannelID
from qupulse.program.loop import Loop
from qupulse.program.waveforms import Waveform
from qupulse.serialization import PulseRegistryType
from qupulse.expressions import ExpressionScalar

from qupulse.pulses.pulse_template import PulseTemplate


class TimeReversalPulseTemplate(PulseTemplate):
    """This pulse template reverses the inner pulse template in time."""

    def __init__(self, inner: PulseTemplate,
                 identifier: Optional[str] = None,
                 registry: PulseRegistryType = None):
        super(TimeReversalPulseTemplate, self).__init__(identifier=identifier)
        self._inner = inner
        self._register(registry=registry)

    def with_time_reversal(self) -> 'PulseTemplate':
        from qupulse.pulses import TimeReversalPT
        if self.identifier:
            return TimeReversalPT(self)
        else:
            return self._inner

    @property
    def parameter_names(self) -> Set[str]:
        return self._inner.parameter_names

    @property
    def measurement_names(self) -> Set[str]:
        return self._inner.measurement_names

    @property
    def duration(self) -> ExpressionScalar:
        return self._inner.duration

    @property
    def defined_channels(self) -> Set['ChannelID']:
        return self._inner.defined_channels

    @property
    def integral(self) -> Dict[ChannelID, ExpressionScalar]:
        return self._inner.integral

    def _internal_create_program(self, *, parent_loop: Loop, **kwargs) -> None:
        inner_loop = Loop()
        self._inner._internal_create_program(parent_loop=inner_loop, **kwargs)
        inner_loop.reverse_inplace()

        parent_loop.append_child(inner_loop)

    def build_waveform(self,
                       *args, **kwargs) -> Optional[Waveform]:
        wf = self._inner.build_waveform(*args, **kwargs)
        if wf is not None:
            return wf.reversed()

    def get_serialization_data(self, serializer=None):
        assert serializer is None, "Old stype serialization not implemented for new class"
        return {
            **super().get_serialization_data(),
            'inner': self._inner
        }

    def _is_atomic(self) -> bool:
        return self._inner._is_atomic()
