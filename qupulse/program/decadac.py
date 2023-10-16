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

        self._channel_voltage_location = {1: 1545,
                                          2: 1561,
                                          3: 1577,
                                          4: 1593,
                                          5: 1609,
                                          6: 1625,
                                          7: 1641,
                                          8: 1657,
                                          9: 1673,
                                          10: 1689,
                                          11: 1705,
                                          12: 1721,
                                          13: 1737,
                                          14: 1753,
                                          15: 1769,
                                          16: 1785,
                                          17: 1801,
                                          18: 1817,
                                          19: 1833,
                                          20: 1849}
        self._safe_memory_locations = {40693, 40694, 40695, 40696, 40697, 40698, 40699, 40700, 40701, 40702, 40703}

        self._block_number = 1
        self._iteration_count = 0
        self._program = '{'

    @classmethod
    def from_channel_dict(cls, channels: Mapping[ChannelID, int]):
        assert len(set(channels.values())) == len(channels), "no duplicate target channels"
        channel_list = [None] * 20
        for ch_name, ch_idx in channels.items():
            channel_list[ch_idx] = ch_name
        return cls(tuple(channel_list))

    def convert_voltage(v) -> int:
        return int((v + 10) / 20 * 65535)

    def inner_scope(self, scope: Scope) -> Scope:
        """This function is necessary to inject program builder specific parameter implementations into the build
        process."""
        raise NotImplementedError('TODO')

    def hold_voltage(self, duration: HardwareTime, voltages: Mapping[ChannelID, HardwareVoltage]):
        command = f'*{self._block_number}:'
        command += 'X' + str(768 + self._block_number) + ';'  # wait for the previous count to finish
        for channel in voltages:
            channel_number = self._name_to_idx(channel)
            dac_value = self.convert_voltage(voltages[channel])
            B_value = int(channel_number / 5)
            C_value = channel_number % 5

            command += f'B{B_value};C{C_value};D{dac_value};'

        duration = int(duration * 1000000)
        command += f'${duration}'

        self._program += command
        self._block_number += 1

    def play_arbitrary_waveform(self, waveform: Waveform):
        raise NotImplementedError('Not implemented yet (postponed)')

    def measure(self, measurements: Optional[Sequence[MeasurementWindow]]):
        """Ignores measurements"""
        pass

    def with_repetition(self, repetition_count: RepetitionCount,
                        measurements: Optional[Sequence[MeasurementWindow]] = None) -> Iterable['ProgramBuilder']:
        command = f'*{self._block_number}:'
        location = str(self._safe_memory_locations[self._iteration_count])
        iteration_command = 'A' + location + ';P' + str(repetition_count) + ';'
        block_command = f'*{self._block_number}:'
        block_command += 'X' + str(768 + self._block_number) + ';'
        command = iteration_command + block_command

    def end_repetition(self):
        location = str(self._safe_memory_locations[self._iteration_count])
        X_command = 1280 + self._block_number + 1
        rep_flag = f'A{location};+-1;X' + str(X_command) + ';'
        self._block_number += 1
        self._iteration_count += 1
        self._program += rep_flag

        # TODO: nested repetitions are impossible with this implementation

    def with_sequence(self,
                      measurements: Optional[Sequence[MeasurementWindow]] = None) -> ContextManager['ProgramBuilder']:
        raise NotImplementedError('TODO')

    def new_subprogram(self, global_transformation: 'Transformation' = None) -> ContextManager['ProgramBuilder']:
        raise NotImplementedError('Not implemented yet (postponed)')

    def with_iteration(self, index_name: str, rng: range,
                       measurements: Optional[Sequence[MeasurementWindow]] = None) -> Iterable['ProgramBuilder']:
        channel_number = self._name_to_idx(index_name)
        dac_value = self.convert_voltage(rng.step)

        command = 'A' + str(self._channel_voltage_location[channel_number]) + ';'  # point to voltage location
        command += '+' + str(dac_value) + ';'
        self._program += command
        # TODO : no way to tell how long each iteration should last

    def to_program(self) -> Optional[DecaDACASCIIProgram]:
        """Further addition of new elements might fail after finalizing the program."""

        # TODO: implement proper triggering
        # TODO: implement additional compression if necessary

        raise NotImplementedError('TODO')
