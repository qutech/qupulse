"""This module defines the common interface for arbitrary waveform generators.

Classes:
    - AWG: Common AWG interface.
    - DummyAWG: A software stub implementation of the AWG interface.
    - ProgramOverwriteException
    - OutOfWaveformMemoryException
"""

from abc import abstractmethod, abstractproperty
from typing import Set, Tuple, List, Callable, Optional

from qctoolkit.utils.types import ChannelID
from qctoolkit.hardware.program import Loop
from qctoolkit.comparable import Comparable
from qctoolkit.pulses.instructions import InstructionSequence

__all__ = ["AWG", "Program", "ProgramOverwriteException",
           "OutOfWaveformMemoryException"]

Program = InstructionSequence


class AWG(Comparable):
    """An arbitrary waveform generator abstraction class.

    It represents a set of channels that have to have(hardware enforced) the same:
        -control flow
        -sample rate

    It keeps track of the AWG state and manages waveforms and programs on the hardware.
    """

    def __init__(self, identifier: str):
        self._identifier = identifier

    @property
    def identifier(self) -> str:
        return self._identifier

    @abstractproperty
    def num_channels(self):
        """Number of channels"""

    @abstractproperty
    def num_markers(self):
        """Number of marker channels"""

    @abstractmethod
    def upload(self, name: str,
               program: Loop,
               channels: Tuple[Optional[ChannelID], ...],
               markers: Tuple[Optional[ChannelID], ...],
               voltage_transformation: Tuple[Optional[Callable], ...],
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

    @abstractmethod
    def remove(self, name: str) -> None:
        """Remove a program from the AWG.

        Also discards all waveforms referenced only by the program identified by name.

        Args:
            name: The name of the program to remove.
        """

    @abstractmethod
    def arm(self, name: str) -> None:
        """Load the program 'name' and arm the device for running it."""

    @abstractproperty
    def programs(self) -> Set[str]:
        """The set of program names that can currently be executed on the hardware AWG."""

    @abstractproperty
    def sample_rate(self) -> float:
        """The sample rate of the AWG."""

    @property
    def compare_key(self) -> int:
        """Comparison and hashing is based on the id of the AWG so different devices with the same properties
        are ot equal"""
        return id(self)

    def __copy__(self) -> None:
        raise NotImplementedError()

    def __deepcopy__(self, memodict={}) -> None:
        raise NotImplementedError()


class ProgramOverwriteException(Exception):

    def __init__(self, name) -> None:
        super().__init__()
        self.name = name

    def __str__(self) -> str:
        return "A program with the given name '{}' is already present on the device." \
               " Use force to overwrite.".format(self.name)


class OutOfWaveformMemoryException(Exception):

    def __str__(self) -> str:
        return "Out of memory error adding waveform to waveform memory."


class ChannelNotFoundException(Exception):
    def __init__(self, channel):
        self.channel = channel

    def __str__(self) -> str:
        return 'Marker or channel not found: {}'.format(self.channel)
