"""This module defines the PulseControlInterface, which offers functionality to convert instruction
sequences obtained by sequencing PulseTemplate structures into MATLAB-interpretable
pulse-control pulse structures."""

from math import floor
from typing import Any, Dict, List, Tuple

import numpy

from qctoolkit.pulses.instructions import Waveform, EXECInstruction, \
    STOPInstruction, InstructionSequence

__all__ = ["PulseControlInterface"]


class PulseControlInterface:
    """Offers functionality to convert instruction sequences obtained by sequencing PulseTemplate
    structures into MATLAB-interpretable pulse control pulse structures."""

    Pulse = Dict[str, Any]
    PulseGroup = Dict[str, Any]

    def __init__(self, sample_rate: int, time_scaling: float=0.001) -> None:
        """Initialize PulseControlInterface.

        Arguments:
            sample_rate (int): The rate in Hz at which waveforms are sampled.
            time_scaling (float): A factor that scales the time domain defined in PulseTemplates.
                (default = 0.001, i.e., one unit of time in a PulseTemplate corresponds to one
                microsecond)
        """
        super().__init__()
        self.__sample_rate = sample_rate
        self.__time_scaling = time_scaling

    @staticmethod
    def __get_waveform_name(waveform: Waveform) -> str:
        # returns a unique identifier for a waveform object
        return 'wf_{}'.format(hash(waveform))

    def create_waveform_struct(self,
                               waveform: Waveform,
                               name: str) -> 'PulseControlInterface.Pulse':
        """Construct a dictionary adhering to the waveform struct definition in pulse control.

        Arguments:
            waveform (Waveform): The Waveform object to convert.
            name (str): Value for the name field in the resulting waveform dictionary.
        Returns:
            a dictionary representing waveform as a waveform struct for pulse control
        """
        sample_count = floor(waveform.duration * self.__time_scaling * self.__sample_rate) + 1
        sample_times = numpy.linspace(0, waveform.duration, sample_count)
        sampled_waveform = waveform.sample(sample_times)
        struct = dict(name=name,
                      data=dict(wf=sampled_waveform.tolist(),
                                marker=numpy.zeros_like(sampled_waveform).tolist(),
                                clk=self.__sample_rate))
        # TODO: how to deal with the second channel expected in waveform structs in pulse control?
        return struct

    def create_pulse_group(self, sequence: InstructionSequence, name: str)\
            -> Tuple['PulseControlInterface.PulseGroup', List['PulseControlInterface.Pulse']]:
        """Construct a dictionary adhering to the pulse group struct definition in pulse control.

        All waveforms in the given InstructionSequence are converted to waveform pulse structs and
        returned as a list in the second component of the returned tuple. The first component of
        the result is a pulse group dictionary denoting the sequence of waveforms using their
        indices in the returned list. create_pulse_group detects multiple use of waveforms and sets
        up the pulse group dictionary accordingly.

        Note that pulses are not registered in pulse control. To achieve this and update the pulse
        group struct accordingly, the dedicated MATLAB script has to be invoked.

        The function will raise an Exception if the given InstructionBlock does contain branching
        instructions, which are not supported by pulse control.

        Arguments:
            sequence (InstructionSequence): The InstructionSequence to convert.
            name (str): Value for the name field in the resulting pulse group dictionary.
        Returns:
            a dictionary representing sequence as pulse group struct for pulse control
        """
        if [x for x in sequence if not isinstance(x, (EXECInstruction, STOPInstruction))]:
            raise Exception("Hardware based branching is not supported by pulse-control.")

        waveforms = [instruction.waveform
                     for instruction in sequence if isinstance(instruction, EXECInstruction)]

        pulse_group = dict(pulses=[],
                           nrep=[],
                           name=name,
                           chan=1,
                           ctrl='notrig')

        waveform_ids = dict()
        waveform_structs = list()

        for waveform in waveforms:
            if waveform not in waveform_ids:
                name = self.__get_waveform_name(waveform)
                waveform_struct = self.create_waveform_struct(waveform, name)
                waveform_structs.append(waveform_struct)
                index = len(waveform_structs) - 1
                waveform_ids[waveform] = index
            else:
                index = waveform_ids[waveform]
            if pulse_group['pulses'] and pulse_group['pulses'][-1] == index:
                pulse_group['nrep'][-1] += 1
            else:
                pulse_group['pulses'].append(index)
                pulse_group['nrep'].append(1)

        return (pulse_group, waveform_structs)
