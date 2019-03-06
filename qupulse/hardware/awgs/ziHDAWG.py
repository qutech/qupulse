from typing import List, Tuple, Set, NamedTuple, Callable, Optional, Any, Sequence, cast, Generator, Union, Dict

from qupulse.hardware.awgs.base import AWG


assert(sys.byteorder == 'little')


class HDAWGRepresentation:
    """docstring for HDAWGRepresentation"""
    def __init__(self, device_id=None, data_server_addr=None, data_server_port=None, external_trigger=False, reset=False):
        """
        :param device_id:         Device id that uniquely identifies this device to the data server
        :param data_server_addr:  Data server address
        :param data_server_port:  Data server port
        :param external_trigger:  Not supported yet
        :param reset:             Reset device before initialization
        """
        # TODO: initialize device here
        self._instr = None

        if external_trigger:
            raise NotImplementedError()  # pragma: no cover

       	if reset:
            raise NotImplementedError()  # pragma: no cover

        self.initialize()

        self._channel_pair_AB = HDAWGChannelPair(self, (1, 2), str(device_id) + '_AB')
        self._channel_pair_CD = HDAWGChannelPair(self, (3, 4), str(device_id) + '_CD')
        self._channel_pair_EF = HDAWGChannelPair(self, (5, 6), str(device_id) + '_EF')
        self._channel_pair_GH = HDAWGChannelPair(self, (7, 8), str(device_id) + '_GH')

    @property
    def channel_pair_AB(self) -> 'HDAWGChannelPair':
        return self._channel_pair_AB

    @property
    def channel_pair_CD(self) -> 'HDAWGChannelPair':
        return self._channel_pair_CD

    @property
    def channel_pair_EF(self) -> 'HDAWGChannelPair':
        return self._channel_pair_EF

    @property
    def channel_pair_GH(self) -> 'HDAWGChannelPair':
        return self._channel_pair_GH

    # TODO: Correct device
    @property
    def main_instrument(self) -> None:
        return self._instr

    def initialize(self) -> None:
        raise NotImplementedError()  # pragma: no cover



class HDAWGChannelPair(AWG):
    """Represents a channel pair of the Zurich Instruments HDAWG as an independent AWG entity.
    It represents a set of channels that have to have(hardware enforced) the same:
        -control flow
        -sample rate

    It keeps track of the AWG state and manages waveforms and programs on the hardware.
    """
    def __init__(self, hdawg_device: HDAWGRepresentation, channels: Tuple[int, int], identifier: str):
        super().__init__(identifier)
        # TODO: add ziDAQ reference of device here
        self._device = None

        if channels not in ((1, 2), (3, 4), (5, 6), (7, 8)):
            raise ValueError('Invalid channel pair: {}'.format(channels))
        self._channels = channels

        self._known_programs = dict()  # type: Dict[str, TaborProgramMemory]

    @property
    def num_channels(self) -> int:
        """Number of channels"""
        return 2

    @property
    def num_markers(self) -> int:
        """Number of marker channels"""
        return 2

    def upload(self, name: str,
               program: Loop,
               channels: Tuple[Optional[ChannelID], Optional[ChannelID]],
               markers: Tuple[Optional[ChannelID], Optional[ChannelID]],
               voltage_transformation: Tuple[Callable, Callable],
               force: bool=False) -> None:
        """Upload a program to the AWG.

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
        raise NotImplementedError()  # pragma: no cover

    def remove(self, name: str) -> None:
        """Remove a program from the AWG.

        Also discards all waveforms referenced only by the program identified by name.

        Args:
            name: The name of the program to remove.
        """
        raise NotImplementedError()  # pragma: no cover

    def clear(self) -> None:
        """Removes all programs and waveforms from the AWG.

        Caution: This affects all programs and waveforms on the AWG, not only those uploaded using qupulse!
        """
        raise NotImplementedError()  # pragma: no cover

    def arm(self, name: str) -> None:
    	"""Load the program 'name' and arm the device for running it. If name is None the awg will "dearm" its current
        program."""
        raise NotImplementedError()  # pragma: no cover

    @property
    def programs(self) -> Set[str]:
        """The set of program names that can currently be executed on the hardware AWG."""
        return set(program.name for program in self._known_programs.keys())

    def sample_rate(self) -> float:
        """The sample rate of the AWG."""
        raise NotImplementedError()  # pragma: no cover