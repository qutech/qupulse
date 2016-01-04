
from abc import ABCMeta, abstractmethod, abstractproperty
from typing import Dict,List, Tuple, Set
from collections import Ordered
import numpy as np
import logging


from qctoolkit.pulses.instructions import InstructionBlock, EXECInstruction

__all__ = ["AWG", "Program", "DummyAWG", "ProgramOverwriteException", "OutOfWaveformMemoryExecption"]

Program = List[InstructionBlock]

class AWG(metaclass = ABCMeta):
    """An arbitrary waveform generator abstraction class. It keeps track of the AWG state and manages waveforms and programs on the hardware."""

    @abstractmethod
    def upload(self, name: str, program: List[InstructionBlock]):
        """Take a name for a program, the program and upload all the necessary waveforms to the AWG hardware. This method should be cheap for programs already on the device and can therefore be used for syncing."""

    @abstractmethod
    def remove(self, name: str, force=False):
        """Take the name of a program and remove it from the AWG, deleting all unneeded waveforms in the process."""

    @abstractproperty
    def programs(self) -> Set[str]:
        """Return the set of program names, that can currently be executed on the hardware AWG."""

    # @abstractmethod
    # def clean(self) -> None:
    #     """Delete all waveforms from the hardware AWG that are not needed by the programs on the machine."""

    @abstractmethod
    def run(self, name) -> None:
        """Load the program 'name' and either arm the device for running it or run it."""




class DummyAWG(AWG):
    """Dummy AWG for debugging purposes."""
    def __init__(self, memory=100):
        self.__programs = {} # contains program names and programs
        self.__waveform_memory = [None for i in memory]
        self.__waveform_indices = {} # dict that maps from waveform hash to memory index
        self.__program_wfs = {} # contains program names and necessary waveforms indices

    def add_waveform(self, waveform):
        try:
            index = self.__waveform_memory.index(None)
        except ValueError:
            raise OutOfWaveformMemoryException
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
            indices = frozenset(add_waveform(block.waveform) for block in exec_blocks)
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

    def programs(self):
        return frozenset(self.__programs.keys())

class ProgramOverwriteException(Exception):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def __str__(self) -> str:
        return "A program with the given name '{}' is already present on the device. Use force to overwrite.".format(self.name)

class OutOfWaveformMemoryException(Exception):
    def __str__(self):
        return "Out of memory error adding waveform to waveform memory."
