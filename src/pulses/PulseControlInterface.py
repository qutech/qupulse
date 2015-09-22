from math import floor
import numpy
import scipy.io
import os.path
from typing import Dict, Any, Callable
from abc import abstractmethod, ABCMeta

from .Sequencer import SequencingHardwareInterface, InstructionBlock
from .Instructions import Waveform, EXECInstruction


class PulseControlWrappingInterface(metaclass=ABCMeta):

    @abstractmethod
    def register_pulse(self, waveform: Waveform) -> int:
        pass


class PulseControlWrapper(PulseControlWrappingInterface):
    # A wrapper for live MATLAB interaction?
    # The idea is to use the PulseControlInterface from a MATLAB workspace and obtaining the pulse structs
    # directly from the create_waveform, create_pulse_group functions. However, I am not yet sure how to do this
    # best and my current MATLAB version (R2015a) seems not to support workspace sharing (or I am doing it wrong).

    def __init__(self) -> None:
        super().__init__()
        #self.engine = matlab.engine.connect_matlab()

    def __del__(self) -> None:
        pass
        #self.engine.quit()

    def register_pulse(self, waveform: Waveform) -> int:
        pass
        #self.engine.


class PulseControlInterface(SequencingHardwareInterface):

    def __init__(self, pulse_registration_function: Callable[[Dict[str, Any]], int], sample_rate: float, time_scaling: float=0.001) -> None:
        """Initialize PulseControlInterface.

        Arguments:
        pulse_registration_function -- A function which registers the pulse in pulse control and returns its id.
        sample_rate -- The rate in Hz at which waveforms are sampled.
        time_scaling -- A factor that scales the time domain defined in PulseTemplates. Defaults to 0.001, meaning
        that one unit of time in a PulseTemplate corresponds to one microsecond.
        """
        super().__init__()
        self.__sample_rate = sample_rate
        self.__time_scaling = time_scaling
        self.__pulse_registration_function = pulse_registration_function

    def __get_waveform_name(self, waveform: Waveform) -> str:
        return 'wf_{}'.format(hash(waveform))

    def register_waveform(self, waveform: Waveform) -> None:
        # Due to recent changes, Waveforms can always be recovered from the EXEC-Instructions.
        # Thus, register_waveform seems to have become obsolete (and with it the whole SequencingHardwareInterface).
        # Simply processing the InstructionBlock obtained from Sequencer seems to be sufficient.
        # However, before removing the Interface entirely, I would like to see whether or not this is still true
        # for real hardware interfaces.
        pass

    def create_waveform_struct(self, waveform: Waveform, name: str) -> Dict[str, Any]:
        """Construct a dictonary adhering to the waveform struct definition in pulse control.

        Arguments:
        waveform -- The Waveform object to convert.
        name -- Value for the name field in the resulting waveform dictionary."""
        sample_count = floor(waveform.duration * self.__time_scaling * self.__sample_rate) + 1
        sample_times = numpy.linspace(0, waveform.duration, sample_count)
        sampled_waveform = waveform.sample(sample_times)
        struct = dict(name=name,
                      data=dict(wf=sampled_waveform,
                                marker=numpy.zeros_like(sampled_waveform),
                                clk=self.__sample_rate))
        # TODO: how to deal with the second channel expected in waveform structs in pulse control?
        return struct

    def create_pulse_group(self, block: InstructionBlock, name: str) -> Dict[str, Any]:
        """Construct a dictonary adhering to the pulse group struct definition in pulse control.

        All waveforms in the given InstructionBlock are converted to waveform pulse structs and registered in
        pulse control with the pulse registration function held by the class. create_pulse_group detects
        multiple use of waveforms and sets up the pulse group dictionary accordingly.

        The function will raise an Exception if the given InstructionBlock does contain branching instructions,
        which are not supported by pulse control.

        Arguments:
        block -- The InstructionBlock to convert.
        name -- Value for the name field in the resulting pulse group dictionary.
        """
        if not all(map(lambda x: isinstance(x, EXECInstruction), block.instructions)):
            raise Exception("Hardware based branching is not supported by pulse-control.")

        waveforms = [instruction.waveform for instruction in block.instructions]
        if not waveforms:
            return ""

        pulse_group = dict(pulses=[],
                           nrep=[],
                           name=name,
                           chan=1,
                           ctrl='notrig')

        registered_waveforms = dict()

        for waveform in waveforms:
            if waveform not in registered_waveforms:
                name = self.__get_waveform_name(waveform)
                waveform_struct = self.create_waveform_struct(waveform, name)
                registered_waveforms[waveform] = self.__pulse_registration_function(waveform_struct)
            if pulse_group['pulses'] and pulse_group['pulses'][-1] == registered_waveforms[waveform]:
                pulse_group['nrep'][-1] += 1
            else:
                pulse_group['pulses'].append(registered_waveforms[waveform])
                pulse_group['nrep'].append(1)

        return pulse_group

    # def create_pulse_group(self, block: InstructionBlock, name: str) -> str:
    #     if not all(map(lambda x: isinstance(x, EXECInstruction), block.instructions)):
    #         raise Exception("Hardware based branching is not supported by pulse-control.")
    #
    #     waveforms = [instruction.waveform for instruction in block.instructions]
    #     if not waveforms:
    #         return ""
    #
    #     pulse_group_code = '% Set up pulse group structure \'{0}\'\r\n' \
    #                        'clear {0};\r\n' \
    #                        '{0}.pulses = [];\r\n' \
    #                        '{0}.nrep = [];\r\n' \
    #                        '{0}.name = \'main\';\r\n' \
    #                        '{0}.chan = 1;\r\n' \
    #                        '{0}.ctrl = \'notrig\';\r\n\r\n'.format(name)
    #
    #     pulse_group_code += '% Load waveforms, register them and add them to the group\r\n'.format(name)
    #     for waveform in waveforms:
    #         waveform_name = self.__get_waveform_name(waveform)
    #         pulse_group_code += 'clear {0};\r\n' \
    #                         '{0} = load(\'{0}\');\r\n' \
    #                         '{0} = plsdefault({0});\r\n' \
    #                         '{1}.pulses(end + 1) = plsreg({0});\r\n' \
    #                         '{1}.nrep(end + 1) = 1;\r\n'.format(waveform_name, name)
    #
    #     return pulse_group_code
