from typing import Any, Dict, List, Set, Optional, Union, Mapping, FrozenSet, cast, Callable
from numbers import Real
import warnings

import sympy

from qupulse.expressions import Expression, ExpressionScalar
from qupulse.serialization import Serializer, PulseRegistryType
from qupulse.parameter_scope import Scope

from qupulse.utils import cached_property
from qupulse.utils.types import ChannelID
from qupulse.pulses.measurement import MeasurementWindow, MeasurementDeclaration
from qupulse.pulses.pulse_template import AtomicPulseTemplate, PulseTemplate
from qupulse._program.waveforms import Waveform, ArithmeticWaveform, TransformingWaveform
from qupulse._program.transformation import Transformation, ScalingTransformation, OffsetTransformation,\
    IdentityTransformation


class ArrayPulseTemplate(AtomicPulseTemplate):
    def __init__(self, channel: ChannelID, array: Union[Expression, Sequence[float]], dt: Expression,
                 identifier: str=None, measurements: Sequence[MeasurementDeclaration], registry: PulseRegistryType=None):
        AtomicPulseTemplate.__init__(self, identifier=identifier, measurements=measurements)
        self._channel = channel
        self._array = array
        self._dt = dt

        self._register(registry=registry)

    def _array


    def build_waveform(self, parameters: Mapping[str, Real], channel_mapping: Dict[ChannelID, Optional[ChannelID]]) -> \
    Optional[Waveform]:
        pass

    def _as_expression(self) -> Dict[ChannelID, ExpressionScalar]:
        pass

    @property
    def parameter_names(self) -> Set[str]:
        pass

    @property
    def duration(self) -> ExpressionScalar:
        pass

    @property
    def defined_channels(self) -> Set['ChannelID']:
        pass

    @property
    def integral(self) -> Dict[ChannelID, ExpressionScalar]:
        pass

