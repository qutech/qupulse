"""This module defines the common interface for arbitrary waveform generators.

Classes:
    - AWG: Common AWG interface.
    - DummyAWG: A software stub implementation of the AWG interface.
    - ProgramOverwriteException
    - OutOfWaveformMemoryException
"""

from abc import ABCMeta, abstractmethod, abstractproperty
from typing import Set, Tuple, List

from qctoolkit.pulses.instructions import InstructionSequence, EXECInstruction

__all__ = ["AWG", "Program", "DummyAWG", "ProgramOverwriteException",
           "OutOfWaveformMemoryExecption"]

Program = InstructionSequence


class AWG(metaclass=ABCMeta):
    """An arbitrary waveform generator abstraction class.

    It keeps track of the AWG state and manages waveforms and programs on the hardware.
    """

    @abstractmethod
    def upload(self, name: str, program: Program, force: bool=False) -> None:
        """Upload a program to the AWG.

        Physically uploads all waveforms required by the program - excluding those already present -
        to the device and sets up playback sequences accordingly.
        This method should be cheap for program already on the device and can therefore be used
        for syncing.

        Args:
            name (str): A name for the program on the AWG.
            program (Program): The program (a sequence of instructions) to upload.
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
    def run(self, name: str) -> None:
        """Load the program 'name' and either arm the device for running it or run it."""
        # todo: isn't this semantically unlcear and should be separated into two explicit methods

    @abstractproperty
    def programs(self) -> Set[str]:
        """The set of program names that can currently be executed on the hardware AWG."""

    @abstractproperty
    def sample_rate(self) -> float:
        """The sample rate of the AWG."""

    @abstractproperty
    def identifier(self) -> str:
        """Return a hardware identifier string."""

    @abstractproperty
    def output_range(self) -> Tuple[float, float]:
        """The minimal/maximal voltage the AWG can produce."""


class DummyAWG(AWG):
    """Dummy AWG for debugging purposes."""

    def __init__(self,
                 memory: int=100,
                 sample_rate: float=10,
                 output_range: Tuple[float, float]=(-5,5)) -> None:
        """Create a new DummyAWG instance.

        Args:
            memory (int): Available memory slots for waveforms. (default = 100)
            sample_rate (float): The sample rate of the dummy. (default = 10)
            output_range (float, float): A (min,max)-tuple of possible output values.
                (default = (-5,5)).
        """
        self.__programs = {} # contains program names and programs
        self.__waveform_memory = [None for i in range(memory)]
        self.__waveform_indices = {} # dict that maps from waveform hash to memory index
        self.__program_wfs = {} # contains program names and necessary waveforms indices
        self.__sample_rate = sample_rate
        self.__output_range = output_range

    def add_waveform(self, waveform) -> int:
        try:
            index = self.__waveform_memory.index(None)
        except ValueError:
            raise OutOfWaveformMemoryException()
        self.__waveform_memory[index] = waveform
        self.__waveform_indices[hash(waveform)] = index
        return index

    def upload(self, name, program, force=False):
        if name in self.programs:
            if not force:
                raise ProgramOverwriteException(name)
            else:
                self.remove(name)
                self.upload(name, program)
        else:
            self.__programs[name] = program
            exec_blocks = filter(lambda x: type(x) == EXECInstruction, program)
            indices = frozenset(self.add_waveform(block.waveform) for block in exec_blocks)
            self.__program_wfs[name] = indices

    def remove(self,name):
        if name in self.programs:
            self.__programs.pop(name)
            self.program_wfs.pop(name)
            self.clean()

    def clean(self):
        necessary_wfs = reduce(lambda acc, s: acc.union(s), self.__program_wfs.values(), set())
        all_wfs = set(self.__waveform_indices.values())
        delete = all_wfs - necessary_wfs
        for index in delete:
            wf = self.__waveform_memory(index)
            self.__waveform_indices.pop(wf)
            self.__waveform_memory = None

    def run(self, name: str) -> None:
        raise NotImplementedError()

    @property
    def programs(self):
        return frozenset(self.__programs.keys())

    @property
    def output_range(self) -> Tuple[float, float]:
        return self.__output_range

    @property
    def identifier(self) -> str:
        return "DummyAWG{0}".format(id(self))

    @property
    def sample_rate(self) -> float:
        return self.__sample_rate


class ProgramOverwriteException(Exception):

    def __init__(self, name):
        super().__init__()
        self.name = name

    def __str__(self) -> str:
        return "A program with the given name '{}' is already present on the device." \
               " Use force to overwrite.".format(self.name)


class OutOfWaveformMemoryException(Exception):
    def __str__(self):
        return "Out of memory error adding waveform to waveform memory."
