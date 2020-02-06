from collections import Callable

from qupulse.hardware.awgs.base import AWGChannelFeature


class ChannelAmplitudeOffsetFeature(AWGChannelFeature):
    def __init__(self, offset_get: Callable[[], float], offset_set: Callable[[float], None],
                 amp_get: Callable[[], float], amp_set: Callable[[float], None]):
        """Storing all callables, to call them if needed below"""
        super().__init__()
        self._get_offset = offset_get
        self._set_offset = offset_set
        self._get_amp = amp_get
        self._set_amp = amp_set

    def get_offset(self) -> float:
        """Forwarding call to callable object, which was provided threw __init__"""
        return self._get_offset()

    def set_offset(self, offset: float) -> None:
        """Forwarding call to callable object, which was provided threw __init__"""
        self._set_offset(offset)

    def get_amplitude(self) -> float:
        """Forwarding call to callable object, which was provided threw __init__"""
        return self._get_amp()

    def set_amplitude(self, amplitude: float) -> None:
        """Forwarding call to callable object, which was provided threw __init__"""
        self._set_amp(amplitude)
