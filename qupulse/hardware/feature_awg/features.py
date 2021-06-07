from abc import ABC, abstractmethod
from typing import Callable, Optional, Set, Tuple, Dict, Union, Any, Mapping
from numbers import Real
from enum import Enum

from qupulse._program._loop import Loop
from qupulse.hardware.feature_awg.base import AWGDeviceFeature, AWGChannelFeature, AWGChannelTupleFeature,\
    AWGChannelTuple
from qupulse.utils.types import ChannelID

try:
    # only required for type annotations
    import pyvisa
except ImportError:
    pyvisa = None


########################################################################################################################
# device features
########################################################################################################################
class SCPI(AWGDeviceFeature, ABC):
    """Represents the ability to communicate via SCPI.

    https://en.wikipedia.org/wiki/Standard_Commands_for_Programmable_Instruments
    """

    def __init__(self, visa: 'pyvisa.resources.MessageBasedResource'):
        super().__init__()
        self._socket = visa

    def send_cmd(self, cmd_str):
        self._socket.write(cmd_str)

    def send_query(self, query_str):
        self._socket.query(query_str)

    def close(self):
        self._socket.close()


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

    @abstractmethod
    def reset(self) -> None:
        """
        Resetting the whole device. A command for resetting is send to the Device, the device is initialized again and
        all channel tuples are cleared.
        """
        raise NotImplementedError()

    @abstractmethod
    def trigger(self) -> None:
        """
        This method triggers a device remotely.
        """
        raise NotImplementedError()


class StatusTable(AWGDeviceFeature, ABC):
    @abstractmethod
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

class ReadProgram(AWGChannelTupleFeature, ABC):
    """Read the currently armed and uploaded program from the device. The returned object is highly device specific."""

    @abstractmethod
    def read_complete_program(self) -> Any:
        raise NotImplementedError()


class VolatileParameters(AWGChannelTupleFeature, ABC):
    """Ability to set the values of parameters which were marked as volatile on program creation."""
    @abstractmethod
    def set_volatile_parameters(self, program_name: str, parameters: Mapping[str, Real]) -> None:
        """Set the values of parameters which were marked as volatile on program creation."""
        raise NotImplementedError()


class RepetitionMode(Enum):
    """Some devices support playing a program indefinitely or only once."""
    # Arm once, trigger once -> infinite repetitions
    INFINITE = "infinite"
    # Arm once, trigger N times -> N playbacks
    AUTO_REARM = "auto_rearm"
    # Arm once, trigger N times -> 1 playback
    ONCE = "once"


class ProgramManagement(AWGChannelTupleFeature, ABC):
    def __init__(self, channel_tuple: 'AWGChannelTuple'):
        super().__init__(channel_tuple=channel_tuple)
        self._default_repetition_mode = RepetitionMode.ONCE

    @abstractmethod
    def upload(self, name: str,
               program: Loop,
               channels: Tuple[Optional[ChannelID], ...],
               marker_channels: Tuple[Optional[ChannelID], ...],
               voltage_transformation: Tuple[Optional[Callable], ...],
               repetition_mode: Union[RepetitionMode, str] = None,
               force: bool = False) -> None:
        """
        Upload a program to the AWG.

        Physically uploads all waveforms required by the program - excluding those already present -
        to the device and sets up playback sequences accordingly.
        This method should be cheap for program already on the device and can therefore be used
        for syncing. Programs that are uploaded should be fast(~1 sec) to arm.

        Raises:
            ValueError: if one of channels, marker_channels, voltage_transformation or repetition_mode is invalid

        Args:
            name: A name for the program on the AWG.
            program: The program (a sequence of instructions) to upload.
            channels: Tuple of length num_channels that ChannelIDs of  in the program to use. Position in the list corresponds to the AWG channel
            marker_channels: List of channels in the program to use. Position in the List in the list corresponds to the AWG channel
            voltage_transformation: transformations applied to the waveforms extracted rom the program. Position
            in the list corresponds to the AWG channel
            repetition_mode: how often the program should be played
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

    @property
    @abstractmethod
    def supported_repetition_modes(self) -> Set[RepetitionMode]:
        """Return set of supported repetition modes in the current configuration."""
        raise NotImplementedError()

    @property
    def default_repetition_mode(self) -> RepetitionMode:
        return self._default_repetition_mode

    @default_repetition_mode.setter
    def default_repetition_mode(self, repetition_mode: RepetitionMode):
        repetition_mode = RepetitionMode(repetition_mode)
        if repetition_mode not in self.supported_repetition_modes:
            raise ValueError(f"The repetition mode {repetition_mode} is not supported by {self._channel_tuple}")
        self._default_repetition_mode = repetition_mode


########################################################################################################################
# channel features
########################################################################################################################

class AmplitudeOffsetHandling(Enum):
    IGNORE_OFFSET = 'ignore_offset'  # Offset is ignored.
    CONSIDER_OFFSET = 'consider_offset'  # Offset is discounted from the waveforms.


class VoltageRange(AWGChannelFeature):
    @property
    @abstractmethod
    def offset(self) -> float:
        """Get offset of AWG channel"""
        raise NotImplementedError()

    @property
    @abstractmethod
    def amplitude(self) -> float:
        """Get amplitude of AWG channel"""
        raise NotImplementedError()

    @property
    @abstractmethod
    def amplitude_offset_handling(self) -> AmplitudeOffsetHandling:
        """
        Gets the amplitude and offset handling of this channel. The amplitude-offset controls if the amplitude and
        offset settings are constant or if these should be optimized by the driver
        """
        raise NotImplementedError()

    @amplitude_offset_handling.setter
    @abstractmethod
    def amplitude_offset_handling(self, amp_offs_handling: Union[str, AmplitudeOffsetHandling]) -> None:
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
