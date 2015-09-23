import numpy as np
from matplotlib import pyplot as plt
from typing import Dict, Tuple

from .Parameter import Parameter
from .Sequencer import Sequencer, SequencingHardwareInterface, SequencingElement
from .Instructions import EXECInstruction, WaveformData, InstructionBlock


# class PlottingWaveform(Waveform):
#
#     def __init__(self, waveform_data: WaveformData) -> None:
#         super().__init__(waveform_data.duration)
#         self.__waveform_data = waveform_data
#
#     @property
#     def waveform_data(self) -> WaveformData:
#         return self.__waveform_data


class Plotter(SequencingHardwareInterface):

    def __init__(self, sample_rate: float=10) -> None:
        super().__init__()
        self.__sample_rate = sample_rate

    def register_waveform(self, waveform: WaveformData) -> None:
        pass

    @property
    def sample_rate(self) -> float:
        return self.__sample_rate

    def render(self, block: InstructionBlock) -> Tuple[np.ndarray, np.ndarray]:
        if not all(map(lambda x: isinstance(x, EXECInstruction), block.instructions)):
            raise NotImplementedError('Can only plot waveforms without branching so far.')

        waveforms = [instruction.waveform for instruction in block.instructions]
        if not waveforms:
            return np.array([0]), np.array([0])
        total_time = sum([waveform.duration for waveform in waveforms])

        sample_count = total_time * self.__sample_rate + 1
        ts = np.linspace(0, total_time, num=sample_count)
        voltages = np.empty_like(ts)
        sample = 0
        for waveform in waveforms:
            voltages[sample:sample + waveform.get_sample_count(self.sample_rate)] = waveform.sample(self.sample_rate)
            sample += waveform.get_sample_count(self.sample_rate) - 1
        return ts, voltages



def plot(pulse: SequencingElement, parameters: Dict[str, Parameter]={}, sample_rate: int=10) -> None: # pragma: no cover
    plotter = Plotter(sample_rate=sample_rate)
    sequencer = Sequencer(plotter)
    sequencer.push(pulse, parameters)
    block = sequencer.build()
    if not sequencer.has_finished():
        raise PlottingNotPossibleException(pulse)
    times, voltages = plotter.render(block)

    # plot!
    f = plt.figure()
    ax = f.add_subplot(111)
    ax.step(times, voltages, where='post')

    # add some margins in the presentation
    plt.plot()
    plt.xlim( -0.5, times[-1] + 0.5)
    plt.ylim(min(voltages) - 0.5, max(voltages) + 0.5)

    f.show()


class PlottingNotPossibleException(Exception):
    def __init__(self, pulse) -> None:
        self.pulse = pulse

    def __str__(self) -> str:
        return "Plotting is not possible. {} can not be rendered for pulses that have branching.".format(self.pulse)

