from abc import ABC, abstractmethod
from typing import Callable, Optional, Set, Tuple, Dict, Union

from qupulse._program._loop import Loop
from qupulse.hardware.awgs.base import AWGDeviceFeature, AWGChannelFeature, AWGChannelTupleFeature
from qupulse.utils.types import ChannelID


########################################################################################################################
# device features
########################################################################################################################

class ChannelSynchronization(AWGDeviceFeature, ABC):
    """This Feature is used to synchronise a certain ammount of channels"""

    @abstractmethod
    def synchronize_channels(self, group_size: int) -> None:
        """
        Synchronize in groups of `group_size` channels. Groups of synchronized channels will be provided as
        AWGChannelTuples.

        Args:
            group_size: Number of channels per channel tuple
        """
        raise NotImplementedError()


class DeviceControl(AWGDeviceFeature, ABC):
    """This feature is used for basic communication with a AWG"""

    # TODO (toCheck): is this Feature ok like this?
    @abstractmethod
    def enable(self) -> None:
        """This method generates the selected output waveform."""
        raise NotImplementedError()

    @abstractmethod
    def abort(self) -> None:
        """This method terminates the current generation of the output waveform."""
        raise NotImplementedError()

    @abstractmethod
    def reset(self) -> None:
        """
        Resetting the whole device. A command for resetting is send to the Device, the device is initialized again and
        all channel tuples are cleared.
        """
        raise NotImplementedError()

    @abstractmethod
    def trigger(self) -> None:
        # TODO (toDo): Docstring missing
        """

        """
        raise NotImplementedError()


class StatusTable(AWGDeviceFeature, ABC):
    def get_status_table(self) -> Dict[str, Union[str, float, int]]:
        """
        Send a lot of queries to the AWG about its settings. A good way to visualize is using pandas.DataFrame

        Returns:
            An ordered dictionary with the results
        """
        raise NotImplementedError()


########################################################################################################################
# channel tuple features
########################################################################################################################

class ProgramManagement(AWGChannelTupleFeature, ABC):
    @abstractmethod
    def upload(self, name: str,
               program: Loop,
               channels: Tuple[Optional[ChannelID], ...],
               markers: Tuple[Optional[ChannelID], ...],
               voltage_transformation: Tuple[Optional[Callable], ...],
               force: bool = False) -> None:
        """
        Upload a program to the AWG.

        Physically uploads all waveforms required by the program - excluding those already present -
        to the device and sets up playback sequences accordingly.
        This method should be cheap for program already on the device and can therefore be used
        for syncing. Programs that are uploaded should be fast(~1 sec) to arm.

        Args:
            name: A name for the program on the AWG.
            program: The program (a sequence of instructions) to upload.
            channels: Tuple of length num_channels that ChannelIDs of  in the program to use. Position in the list corresponds to the AWG channel
            markers: List of channels in the program to use. Position in the List in the list corresponds to the AWG channel
            voltage_transformation: transformations applied to the waveforms extracted rom the program. Position
            in the list corresponds to the AWG channel
            force: If a different sequence is already present with the same name, it is
                overwritten if force is set to True. (default = False)
        """
        raise NotImplementedError()

    @abstractmethod
    def remove(self, name: str) -> None:
        """
        Remove a program from the AWG.

        Also discards all waveforms referenced only by the program identified by name.

        Args:
            name: The name of the program to remove.
        """
        raise NotImplementedError()

    @abstractmethod
    def clear(self) -> None:
        """
        Removes all programs and waveforms from the AWG.

        Caution: This affects all programs and waveforms on the AWG, not only those uploaded using qupulse!
        """
        raise NotImplementedError()

    @abstractmethod
    def arm(self, name: Optional[str]) -> None:
        """
        Load the program 'name' and arm the device for running it. If name is None the awg will "dearm" its current
        program.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def programs(self) -> Set[str]:
        """The set of program names that can currently be executed on the hardware AWG."""
        raise NotImplementedError()

    @abstractmethod
    def run_current_program(self) -> None:
        """This method starts running the active program"""
        raise NotImplementedError()

    @abstractmethod
    def change_armed_program(self) -> None:
        raise NotImplementedError()


########################################################################################################################
# channel features
########################################################################################################################

class AmplitudeOffsetHandling:
    IGNORE_OFFSET = 'ignore_offset'  # Offset is ignored.
    CONSIDER_OFFSET = 'consider_offset'  # Offset is discounted from the waveforms.

    _valid = (IGNORE_OFFSET, CONSIDER_OFFSET)


class OffsetAmplitude(AWGChannelFeature):
    @property
    @abstractmethod
    def offset(self) -> float:
        """Get offset of AWG channel"""
        raise NotImplementedError()

    @offset.setter
    @abstractmethod
    def offset(self, offset: float) -> None:
        """Set offset for AWG channel"""
        raise NotImplementedError()

    @property
    @abstractmethod
    def amplitude(self) -> float:
        """Get amplitude of AWG channel"""
        raise NotImplementedError()

    @amplitude.setter
    @abstractmethod
    def amplitude(self, amplitude: float) -> None:
        """Set amplitude for AWG channel"""
        raise NotImplementedError()

    @property
    @abstractmethod
    def amplitude_offset_handling(self) -> str:
        """
        Gets the amplitude and offset handling of this channel. The amplitude-offset controls if the amplitude and
        offset settings are constant or if these should be optimized by the driver
        """
        raise NotImplementedError()

    @amplitude_offset_handling.setter
    @abstractmethod
    def amplitude_offset_handling(self, amp_offs_handling: str) -> None:
        """
        amp_offs_handling: See possible values at `AWGAmplitudeOffsetHandling`
        """
        raise NotImplementedError()


class ActivatableChannels(AWGChannelFeature):
    @property
    @abstractmethod
    def enabled(self) -> bool:
        """
        Returns the the state a channel has at the moment. A channel is either activated or deactivated
        """
        raise NotImplementedError()

    @abstractmethod
    def enable(self):
        """Enables the output of a certain channel"""
        raise NotImplementedError()

    @abstractmethod
    def disable(self):
        """Disables the output of a certain channel"""
        raise NotImplementedError()