from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, ContextManager, Iterable, Tuple

from qupulse import ChannelID, MeasurementWindow
from qupulse.parameter_scope import Scope
from qupulse.program import ProgramBuilder, HardwareTime, HardwareVoltage, Waveform, RepetitionCount, TimeType


@dataclass
class DecaDACASCIIProgram:
    script: str
    duration: TimeType


class DecaDACASCIIBuilder:
    def __init__(self, channels: Tuple[Optional[ChannelID], ...]):
        assert len(channels) in (20,), "Only 5 slots are supported for now"
        self._name_to_idx = {idx: name for idx, name in enumerate(channels) if name is not None}
        self._idx_to_name = channels

    @classmethod
    def from_channel_dict(cls, channels: Mapping[ChannelID, int]):
        assert len(set(channels.values())) == len(channels), "no duplicate target channels"
        channel_list = [None] * 20
        for ch_name, ch_idx in channels.items():
            channel_list[ch_idx] = ch_name
        return cls(tuple(channel_list))

    def inner_scope(self, scope: Scope) -> Scope:
        """This function is necessary to inject program builder specific parameter implementations into the build
        process."""
        raise NotImplementedError('TODO')

    def hold_voltage(self, duration: HardwareTime, voltages: Mapping[ChannelID, HardwareVoltage]):
        raise NotImplementedError('TODO')

    def play_arbitrary_waveform(self, waveform: Waveform):
        raise NotImplementedError('Not implemented yet (postponed)')

    def measure(self, measurements: Optional[Sequence[MeasurementWindow]]):
        """Ignores measurements"""
        pass

    def with_repetition(self, repetition_count: RepetitionCount,
                        measurements: Optional[Sequence[MeasurementWindow]] = None) -> Iterable['ProgramBuilder']:
        raise NotImplementedError('TODO')

    def with_sequence(self,
                      measurements: Optional[Sequence[MeasurementWindow]] = None) -> ContextManager['ProgramBuilder']:
        raise NotImplementedError('TODO')

    def new_subprogram(self, global_transformation: 'Transformation' = None) -> ContextManager['ProgramBuilder']:
        raise NotImplementedError('Not implemented yet (postponed)')

    def with_iteration(self, index_name: str, rng: range,
                       measurements: Optional[Sequence[MeasurementWindow]] = None) -> Iterable['ProgramBuilder']:
        raise NotImplementedError('TODO')

    def to_program(self) -> Optional[DecaDACASCIIProgram]:
        """Further addition of new elements might fail after finalizing the program."""
        raise NotImplementedError('TODO')
