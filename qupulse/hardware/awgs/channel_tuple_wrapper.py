from typing import Tuple, Optional, Callable, Set

from qupulse._program._loop import Loop
from qupulse.hardware.awgs.base import AWGChannelTuple
from qupulse.hardware.awgs.old_base import AWG


class ChannelTupleAdapter(AWG):
    # TODO (toCheck): is this DocString okay like this?
    """
    This class serves as an adapter between the old Class AWG and the new driver abstraction. It routes all the methods
    the AWG class to the corresponding methods of the new driver.
    """

    def __copy__(self) -> None:
        pass

    def __init__(self, channel_tuple: AWGChannelTuple):
        self._channel_tuple = channel_tuple

    def identifier(self) -> str:
        return self._channel_tuple.name

    def num_channels(self) -> int:
        return self._channel_tuple.num_channels

    def num_markers(self) -> int:
        return self._channel_tuple.num_markers

    def upload(self, name: str,
               program: Loop,
               channels: Tuple[Optional["ChannelID"], ...],
               markers: Tuple[Optional["ChannelID"], ...],
               voltage_transformation: Tuple[Optional[Callable], ...],
               force: bool = False) -> None:
        from qupulse.hardware.awgs.tabor import TaborProgramManagement
        return self._channel_tuple[TaborProgramManagement].upload(name, program, channels, markers,
                                                                  voltage_transformation, force)

    def remove(self, name: str) -> None:
        from qupulse.hardware.awgs.tabor import TaborProgramManagement
        return self._channel_tuple[TaborProgramManagement].remove(name)

    def clear(self) -> None:
        from qupulse.hardware.awgs.tabor import TaborProgramManagement
        return self._channel_tuple[TaborProgramManagement].clear()

    def arm(self, name: Optional[str]) -> None:
        from qupulse.hardware.awgs.tabor import TaborProgramManagement
        return self._channel_tuple[TaborProgramManagement].arm(name)

    def programs(self) -> Set[str]:
        from qupulse.hardware.awgs.tabor import TaborProgramManagement
        return self._channel_tuple[TaborProgramManagement].programs

    def sample_rate(self) -> float:
        return self._channel_tuple.sample_rate
