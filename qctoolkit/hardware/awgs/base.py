"""This module defines the common interface for arbitrary waveform generators.

Classes:
    - AWG: Common AWG interface.
    - DummyAWG: A software stub implementation of the AWG interface.
    - ProgramOverwriteException
    - OutOfWaveformMemoryException
"""

from abc import abstractmethod, abstractproperty
from typing import Set, Tuple, List, Callable, Optional

from qctoolkit import ChannelID
from qctoolkit.hardware.program import Loop
from qctoolkit.comparable import Comparable
from qctoolkit.pulses.instructions import InstructionSequence

__all__ = ["AWG", "Program", "DummyAWG", "ProgramOverwriteException",
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
        self.identifier = identifier

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
            name (str): A name for the program on the AWG.
            program (Loop): The program (a sequence of instructions) to upload.
            channels (List): Tuple of length num_channels that ChannelIDs of  in the program to use. Position in the list corresponds to the AWG channel
            markers (List): List of channels in the program to use. Position in the List in the list corresponds to the AWG channel
            voltage_transformation (List): transformations applied to the waveforms extracted rom the program. Position
            in the list corresponds to the AWG channel
            force (bool): If a different sequence is already present with the same name, it is
                overwritten if force is set to True. (default = False)
        """

    @abstractmethod
    def remove(self, name: str) -> None:
        """Remove a program from the AWG.

        Also discards all waveforms referenced only by the program identified by name.

        Args:
            name (str): The name of the program to remove.
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


class DummyAWG(AWG):
    """Dummy AWG for debugging purposes."""

    def __init__(self,
                 memory: int=100,
                 sample_rate: float=10,
                 output_range: Tuple[float, float]=(-5, 5),
                 num_channels: int=1,
                 num_markers: int=1) -> None:
        """Create a new DummyAWG instance.

        Args:
            memory (int): Available memory slots for waveforms. (default = 100)
            sample_rate (float): The sample rate of the dummy. (default = 10)
            output_range (float, float): A (min,max)-tuple of possible output values.
                (default = (-5,5)).
        """
        self._programs = {} # contains program names and programs
        self._waveform_memory = [None for i in range(memory)]
        self._waveform_indices = {} # dict that maps from waveform hash to memory index
        self._program_wfs = {} # contains program names and necessary waveforms indices
        self._sample_rate = sample_rate
        self._output_range = output_range
        self._num_channels = num_channels
        self._num_markers = num_markers
        self._channels = ('default',)
        self._armed = None

    def upload(self, name, program, channels, markers, voltage_transformation, force=False) -> None:
        if name in self.programs:
            if not force:
                raise ProgramOverwriteException(name)
            else:
                self.remove(name)
                self.upload(name, program)
        else:
            self._programs[name] = (program, channels, markers, voltage_transformation)

    def remove(self, name) -> None:
        if name in self.programs:
            self._programs.pop(name)
            self.program_wfs.pop(name)
            self.clean()

    def arm(self, name: str) -> None:
        self._armed = name

    @property
    def programs(self) -> Set[str]:
        return frozenset(self._programs.keys())

    @property
    def output_range(self) -> Tuple[float, float]:
        return self._output_range

    @property
    def identifier(self) -> str:
        return "DummyAWG{0}".format(id(self))

    @property
    def sample_rate(self) -> float:
        return self._sample_rate

    @property
    def num_channels(self):
        return self._num_channels

    @property
    def num_markers(self):
        return self._num_markers


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
