from typing import Tuple, Optional, Callable, Set

from qupulse import ChannelID
from qupulse._program._loop import Loop
from qupulse.hardware.feature_awg.base import AWGChannelTuple
from qupulse.hardware.feature_awg.features import ProgramManagement, VolatileParameters
from qupulse.hardware.awgs.base import AWG


class ChannelTupleAdapter(AWG):
    """
    This class serves as an adapter between the old Class AWG and the new driver abstraction. It routes all the methods
    the AWG class to the corresponding methods of the new driver.
    """
    def __copy__(self) -> None:
        pass

    def __init__(self, channel_tuple: AWGChannelTuple):
        super().__init__(channel_tuple.name)
        self._channel_tuple = channel_tuple

    @property
    def num_channels(self) -> int:
        return len(self._channel_tuple.channels)

    @property
    def num_markers(self) -> int:
        return len(self._channel_tuple.marker_channels)

    def upload(self, name: str,
               program: Loop,
               channels: Tuple[Optional[ChannelID], ...],
               markers: Tuple[Optional[ChannelID], ...],
               voltage_transformation: Tuple[Optional[Callable], ...],
               force: bool = False) -> None:
        return self._channel_tuple[ProgramManagement].upload(name=name, program=program,
                                                             channels=channels,
                                                             marker_channels=markers,
                                                             voltage_transformation=voltage_transformation,
                                                             repetition_mode=None,
                                                             force=force)

    def remove(self, name: str) -> None:
        return self._channel_tuple[ProgramManagement].remove(name)

    def clear(self) -> None:
        return self._channel_tuple[ProgramManagement].clear()

    def arm(self, name: Optional[str]) -> None:
        return self._channel_tuple[ProgramManagement].arm(name)

    @property
    def programs(self) -> Set[str]:
        return self._channel_tuple[ProgramManagement].programs

    @property
    def sample_rate(self) -> float:
        return self._channel_tuple.sample_rate

    def set_volatile_parameters(self, program_name: str, parameters):
        self._channel_tuple[VolatileParameters].set_volatile_parameters(program_name, parameters)

