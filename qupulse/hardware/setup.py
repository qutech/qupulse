from typing import NamedTuple, Set, Callable, Dict, Tuple, Union, Iterable, Any, Mapping
from collections import defaultdict
import warnings
import numbers

from qupulse.hardware.awgs.base import AWG
from qupulse.hardware.dacs import DAC
from qupulse._program._loop import Loop

from qupulse.utils.types import ChannelID

import numpy as np


__all__ = ['PlaybackChannel', 'MarkerChannel', 'HardwareSetup']


class MeasurementMask:
    def __init__(self, dac: DAC, mask_name: str):
        self.dac = dac
        self.mask_name = mask_name

    def __iter__(self):
        yield self.dac
        yield self.mask_name


class _SingleChannel:
    """An actual hardware channel"""
    def __init__(self, awg: AWG, channel_on_awg: int):
        self.awg = awg
        """The AWG the channel is defined on"""

        self.channel_on_awg = channel_on_awg
        """The channel's index(starting with 0) on the AWG."""

    @property
    def compare_key(self) -> Tuple[Any]:
        return (id(self.awg), self.channel_on_awg, type(self))

    def __hash__(self):
        return hash(self.compare_key)

    def __eq__(self, other):
        if not isinstance(other, _SingleChannel): return False
        return self.compare_key == other.compare_key


class PlaybackChannel(_SingleChannel):
    """A hardware channel that is not a marker"""
    def __init__(self, awg: AWG, channel_on_awg: int,
                 voltage_transformation: Callable[[np.ndarray], np.ndarray]=lambda x: x):
        if channel_on_awg >= awg.num_channels:
            raise ValueError('Can not create PlayBack channel {}. AWG only has {} channels'.format(channel_on_awg,
                                                                                                   awg.num_channels))
        super().__init__(awg=awg, channel_on_awg=channel_on_awg)

        self.voltage_transformation = voltage_transformation
        """A transformation that is applied to the pulses on the channel. One use case is to scale up the voltage if an
        amplifier is inserted."""


class MarkerChannel(_SingleChannel):
    """A hardware channel that can only take two values"""
    def __init__(self, awg: AWG, channel_on_awg: int):
        if channel_on_awg >= awg.num_markers:
            raise ValueError('Can not create MarkerBack channel {}. AWG only has {} channels'.format(channel_on_awg,
                                                                                                     awg.num_markers))
        super().__init__(awg=awg, channel_on_awg=channel_on_awg)


RegisteredProgram = NamedTuple('RegisteredProgram', [('program', Loop),
                                                     ('measurement_windows', Dict[str, Tuple[float, float]]),
                                                     ('run_callback', Callable),
                                                     ('awgs_to_upload_to', Set[AWG]),
                                                     ('dacs_to_arm', Set[DAC])])


class HardwareSetup:
    """Representation of the hardware setup.

    The class takes an instruction block, forms it into possibly channel dependent programs
    and registers the programs at the AWGs which modify their program to fit to their capabilities. The class also
    extracts the measurement windows(with absolute times) and hands them over to the DACs which will do further
    processing."""
    def __init__(self):
        self._channel_map = dict()  # type: Dict[ChannelID, Set[_SingleChannel]]

        self._measurement_map = dict()  # type: Dict[str, Set[MeasurementMask]]

        self._registered_programs = dict()  # type: Dict[str, RegisteredProgram]

    def register_program(self, name: str,
                         program: Loop,
                         run_callback=lambda: None, update=False) -> None:
        if not callable(run_callback):
            raise TypeError('The provided run_callback is not callable')

        channels = next(program.get_depth_first_iterator()).waveform.defined_channels
        if channels - set(self._channel_map.keys()):
            raise KeyError('The following channels are unknown to the HardwareSetup: {}'.format(
                channels - set(self._channel_map.keys())))

        temp_measurement_windows = defaultdict(list)
        for mw_name, begins_lengths in program.get_measurement_windows().items():
            temp_measurement_windows[mw_name].append(begins_lengths)

        if set(temp_measurement_windows.keys()) - set(self._measurement_map.keys()):
            raise KeyError('The following measurements are not registered: {}\nUse set_measurement for that.'.format(
                set(temp_measurement_windows.keys()) - set(self._measurement_map.keys())
            ))

        measurement_windows = dict()
        while temp_measurement_windows:
            mw_name, begins_lengths_deque = temp_measurement_windows.popitem()

            begins, lengths = zip(*begins_lengths_deque)
            measurement_windows[mw_name] = (
                np.concatenate(begins),
                np.concatenate(lengths)
            )

        affected_dacs = defaultdict(dict)
        for measurement_name, begins_lengths in measurement_windows.items():
            for dac, mask_name in self._measurement_map[measurement_name]:
                affected_dacs[dac][mask_name] = begins_lengths

        handled_awgs = set()
        awgs_to_channel_info = dict()

        def get_default_info(awg):
            return ([None] * awg.num_channels,
                    [None] * awg.num_channels,
                    [None] * awg.num_markers)

        for channel_id in channels:
            for single_channel in self._channel_map[channel_id]:
                playback_ids, voltage_trafos, marker_ids = \
                        awgs_to_channel_info.setdefault(single_channel.awg, get_default_info(single_channel.awg))

                if isinstance(single_channel, PlaybackChannel):
                    playback_ids[single_channel.channel_on_awg] = channel_id
                    voltage_trafos[single_channel.channel_on_awg] = single_channel.voltage_transformation
                elif isinstance(single_channel, MarkerChannel):
                    marker_ids[single_channel.channel_on_awg] = channel_id

        for awg, (playback_ids, voltage_trafos, marker_ids) in awgs_to_channel_info.items():
            if awg in handled_awgs:
                raise ValueError('AWG has two programs')
            else:
                handled_awgs.add(awg)
            awg.upload(name,
                       program=program,
                       channels=tuple(playback_ids),
                       markers=tuple(marker_ids),
                       force=update,
                       voltage_transformation=tuple(voltage_trafos))

        for dac, dac_windows in affected_dacs.items():
            dac.register_measurement_windows(name, dac_windows)

        self._registered_programs[name] = RegisteredProgram(program=program,
                                                            measurement_windows=measurement_windows,
                                                            run_callback=run_callback,
                                                            awgs_to_upload_to=handled_awgs,
                                                            dacs_to_arm=set(affected_dacs.keys()))

    def remove_program(self, name: str):
        if name in self._registered_programs:
            program_info = self._registered_programs.pop(name)
            for awg in program_info.awgs_to_upload_to:
                try:
                    awg.arm(None)
                    awg.remove(name)
                except RuntimeError:
                    warnings.warn("Could not remove Program({}) from AWG({})".format(name, awg.identifier))

            for dac in program_info.dacs_to_arm:
                try:
                    dac.delete_program(name)
                except RuntimeError:
                    warnings.warn("Could not remove Program({}) from DAC({})".format(name, dac))

    def clear_programs(self) -> None:
        """Clears all programs from all known AWG and DAC devices.

        Does not affect channel configurations or measurement masks set by set_channel or set_measurement."""
        for awg in self.known_awgs:
            awg.clear()
        for dac in self.known_dacs:
            dac.clear()
        self._registered_programs = dict()

    @property
    def known_awgs(self) -> Set[AWG]:
        return {single_channel.awg
                for single_channel_set in self._channel_map.values()
                for single_channel in single_channel_set}

    @property
    def known_dacs(self) -> Set[DAC]:
        masks = set.union(*self._measurement_map.values()) if self._measurement_map else set()
        dacs = {mask.dac for mask in masks}
        return dacs

    def arm_program(self, name: str) -> None:
        """Assert program is in memory. Hardware will wait for trigger event"""
        if name not in self._registered_programs:
            raise KeyError('{} is not a registered program'.format(name))

        *_, awgs_to_upload_to, dacs_to_arm = self._registered_programs[name]
        for awg in self.known_awgs:
            if awg in awgs_to_upload_to:
                awg.arm(name)
            else:
                # The other AWGs should ignore the trigger
                awg.arm(None)
        for dac in dacs_to_arm:
            dac.arm_program(name)

    def run_program(self, name) -> None:
        """Calls arm program and starts it using the run callback"""
        self.arm_program(name)
        self._registered_programs[name].run_callback()

    def set_channel(self, identifier: ChannelID,
                    single_channel: Union[_SingleChannel, Iterable[_SingleChannel]],
                    allow_multiple_registration: bool=False) -> None:
        if isinstance(single_channel, (PlaybackChannel, MarkerChannel)):
            if identifier in self._channel_map:
                warnings.warn(
                    "You add a single hardware channel to an already existing channel id. This is deprecated and will be removed in a future version. Please add all channels at once.",
                    DeprecationWarning)
                single_channel = self._channel_map[identifier] | {single_channel}
            else:
                single_channel = {single_channel}
        else:
            try:
                single_channel = set(single_channel)
            except TypeError:
                raise TypeError('Channel must be (a list of) either a playback or a marker channel')

        if not allow_multiple_registration:
            for ch_id, channel_set in self._channel_map.items():
                if single_channel & channel_set:
                    raise ValueError('Channel already registered as {} for channel {}'.format(
                        type(self._channel_map[ch_id]).__name__, ch_id))

        for s_channel in single_channel:
            if not isinstance(s_channel, (PlaybackChannel, MarkerChannel)):
                raise TypeError('Channel must be (a list of) either a playback or a marker channel')

        self._channel_map[identifier] = single_channel

    def set_measurement(self, measurement_name: str,
                        measurement_mask: Union[MeasurementMask, Iterable[MeasurementMask]],
                        allow_multiple_registration: bool=False):
        if isinstance(measurement_mask, MeasurementMask):
            if measurement_name in self._measurement_map:
                warnings.warn(
                    "You add a measurement mask to an already registered measurement name. This is deprecated and will be removed in a future version. Please add all measurement masks at once.",
                    DeprecationWarning)
                measurement_mask = self._measurement_map[measurement_name] | {measurement_mask}
            else:
                measurement_mask = {measurement_mask}
        else:
            try:
                measurement_mask = set(measurement_mask)
            except TypeError:
                raise TypeError('Mask must be (a list) of type MeasurementMask')

        if not allow_multiple_registration:
            for old_measurement_name, mask_set in self._measurement_map.items():
                if measurement_mask & mask_set:
                    raise ValueError('Measurement mask already registered for measurement "{}"'.format(old_measurement_name))

        self._measurement_map[measurement_name] = measurement_mask

    def rm_channel(self, identifier: ChannelID) -> None:
        self._channel_map.pop(identifier)

    def registered_channels(self) -> Dict[ChannelID, Set[_SingleChannel]]:
        return self._channel_map

    def update_parameters(self, name: str, parameters: Mapping[str, numbers.Real]):
        *_, awgs, dacs = self._registered_programs[name]

        for awg in self.known_awgs:
            if awg in awgs:
                awg.set_volatile_parameters(name, parameters)

    @property
    def registered_programs(self) -> Dict[str, RegisteredProgram]:
        return self._registered_programs








