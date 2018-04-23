from abc import abstractmethod
from typing import Tuple, Dict, Any

import numpy as np

from qctoolkit.utils.types import DocStringABCMeta, ChannelID
from qctoolkit.pulses.instructions import InstructionBlock, Waveform, AbstractInstructionBlock, \
    ImmutableInstructionBlock, EXECInstruction, REPJInstruction, GOTOInstruction, STOPInstruction, InstructionPointer, \
    Instruction


class ChannelDistortionCompensator(metaclass=DocStringABCMeta):
    """Convolution based channel distortion compensation model"""

    @property
    @abstractmethod
    def input_window_width(self) -> int:
        """Sample count in input window."""

    @abstractmethod
    def apply(self, input_samples: np.ndarray) -> np.float:
        """Apply compensation filter to window of samples and return result."""

    @property
    @abstractmethod
    def pre_padding(self) -> np.ndarray:
        """Dummy sample values to be prepended to the first samples of a waveform. Length: input_window_width."""

    @property
    @abstractmethod
    def post_padding(self) -> np.ndarray:
        """Dummy sample values to be appended to the last samples of a waveform. Length: input_window_width."""


class ControlFlowEmulator:

    def __init__(self, program: ImmutableInstructionBlock):
        self.__ip = InstructionPointer(program, 0)
        self.__repj_counters = {}

    def make_step(self):
        instruction = self.__ip.block[self.__ip.offset]
        if isinstance(instruction, STOPInstruction):
            return
        self.__ip = InstructionPointer(self.__ip.block, self.__ip.offset + 1) # default increment

        if isinstance(instruction, REPJInstruction):
            if instruction not in self.__repj_counters:
                self.__repj_counters[instruction] = instruction.count
            if self.__repj_counters[instruction] > 0:
                self.__repj_counters[instruction] -= 1
                self.__ip = instruction.target
            else:
                self.__repj_counters.pop(instruction) # clean up and go on
        elif isinstance(instruction, GOTOInstruction):
            self.__ip = instruction.target

    @property
    def instruction_pointer(self) -> InstructionPointer:
        return InstructionPointer(self.__ip.block, self.__ip.offset)

    @property
    def instruction(self) -> Instruction:
        return self.__ip.block[self.__ip.offset]


# class MultiChannelBuffer:
#
#     def __init__(self, channel_ids):
#         self.__channel_ids = frozenset(channel_ids)
#         self.
#
#     @property
#     def channel_ids(self):
#         return self.__channel_ids
#
#     def __getitem__(self, item):
#         return self.



class DistortionCompensator:

    def __init__(self,
                 channel_compensators: Dict[ChannelID, ChannelDistortionCompensator],
                 channel_frequencies: Dict[ChannelID, float]) -> None:
        self.__compensators = channel_compensators
        self.__channel_frequencies = channel_frequencies
        self.__buffers = {}
        self.__output_buffers = {}

    def gen_samples(self, start, duration, channel):
        return np.linspace(start, start+duration, duration * self.__channel_frequencies[channel], endpoint=False)

    def buffer_waveform(self, wf: Waveform, t: float) -> Dict[np.ndarray]:
        buffers = {}
        for channel_id in wf.defined_channels:
            t0 = np.ceil(t * self.__channel_frequencies[channel_id]) * self.__channel_frequencies[channel_id]
            t_off = t0 - t # offset of first sample in waveform
            samples = self.gen_samples(t_off, wf.duration, channel_id) # get sample times in waveform time frame
            buffers[channel_id] = wf.unsafe_sample(channel_id, samples)
            #self.__buffers[channel_id] = np.concatenate((self.__buffers[channel_id],
            #                                             wf.unsafe_sample(channel_id, samples)))

    def consume_buffer(self) -> Dict[np.ndarray]:
        # if current buffer is too small to process, consume_buffer will do nothing
        output_buffers = {}
        for channel_id in self.__buffers:
            width = self.__compensators[channel_id].input_window_width
            count = len(self.__buffers[channel_id]) - (width - 1)
            output_buffers[channel_id] = np.zeros(count)
            for i in range(len(output_buffers[channel_id])):
                output_buffers[i] = self.__compensators[channel_id].apply(self.__buffers[channel_id][i:i+width])
            self.__buffers[channel_id] = self.__buffers[channel_id][-(width - 1):]
        return output_buffers

    def compensate(self, program: AbstractInstructionBlock) -> Dict[ChannelID, np.ndarray]:
        emulator = ControlFlowEmulator(program)
        channel_ids = program.instructions.defined_channels
        self.__buffers = {channel_id: np.copy(self.__compensators[channel_id].pre_padding)
                          for channel_id in channel_ids}
        output_buffers = {channel_id: np.zeros(0) for channel_id in channel_ids}
        t = 0
        while not isinstance(emulator.instruction, STOPInstruction):
            while not (isinstance(emulator.instruction, EXECInstruction) or isinstance(emulator.instruction, STOPInstruction)):
                emulator.make_step()
            if isinstance(emulator.instruction, EXECInstruction):
                wf = emulator.instruction.waveform
                sampled_waveforms = self.buffer_waveform(wf, t)
                t += wf.duration

                for channel_id in channel_ids:
                    self.__buffers[channel_id] = np.concatenate((self.__buffers[channel_id],
                                                                 sampled_waveforms[channel_id]))
                compensated_waveforms = self.consume_buffer()
                for channel_id in channel_ids:
                    output_buffers[channel_id] = np.concatenate((output_buffers[channel_id],
                                                                 compensated_waveforms[channel_id]))

        for channel_id in channel_ids:
            self.__buffers[channel_id] = np.concatenate((self.__buffers[channel_id],
                                                         self.__compensators[channel_id].post_padding))
        compensated_waveforms = self.consume_buffer()
        for channel_id in channel_ids:
            output_buffers[channel_id] = np.concatenate((output_buffers[channel_id],
                                                         compensated_waveforms[channel_id]))

        return output_buffers





