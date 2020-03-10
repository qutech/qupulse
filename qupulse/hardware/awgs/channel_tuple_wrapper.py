from typing import Tuple, Optional, Callable, Set

from qupulse._program._loop import Loop
from qupulse.hardware.awgs.old_base import AWG


class ChannelTupleAdapter(AWG):
    def __copy__(self) -> None:
        pass

    def __init__(self, method_identifier, method_num_channels, method_num_marker, method_upload, method_remove,
                 method_clear, method_arm, method_programs, method_sample_rate):
        self.m_identifier = method_identifier
        self.m_num_channels = method_num_channels
        self.m_num_marker = method_num_marker
        self.m_upload = method_upload
        self.m_remove = method_remove
        self.m_clear = method_clear
        self.m_arm = method_arm
        self.m_programs = method_programs
        self.m_sample_rate = method_sample_rate

    def identifier(self) -> str:
        return self.m_identifier

    def num_channels(self) -> int:
        return self.m_num_channels()

    def num_markers(self) -> int:
        return self.m_num_marker

    def upload(self, name: str,
               program: Loop,
               channels: Tuple[Optional["ChannelID"], ...],
               markers: Tuple[Optional["ChannelID"], ...],
               voltage_transformation: Tuple[Optional[Callable], ...],
               force: bool = False) -> None:
        return self.m_upload(name, program, channels, markers, voltage_transformation, force)

    def remove(self, name: str) -> None:
        return self.m_remove(name)

    def clear(self) -> None:
        return self.m_clear

    def arm(self, name: Optional[str]) -> None:
        return self.m_arm(name)

    def programs(self) -> Set[str]:
        return self.m_programs

    def sample_rate(self) -> float:
        return self.m_sample_rate
