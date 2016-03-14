import numpy as np
from matplotlib import pyplot as plt
from typing import Dict, Tuple

from .parameters import Parameter
from .sequencing import Sequencer, SequencingElement
from .instructions import EXECInstruction, STOPInstruction, InstructionSequence


__all__ = ["Plotter", "plot", "PlottingNotPossibleException"]


class Plotter:

    def __init__(self, sample_rate: float=10) -> None:
        super().__init__()
        self.__sample_rate = sample_rate

    def render(self, sequence: InstructionSequence) -> Tuple[np.ndarray, np.ndarray]:
        if not all(map(lambda x: isinstance(x, (EXECInstruction, STOPInstruction)), sequence)):
            raise NotImplementedError('Can only plot waveforms without branching so far.')

        waveforms = [instruction.waveform for instruction in sequence if isinstance(instruction, EXECInstruction)]
        if not waveforms:
            return [], []
        total_time = sum([waveform.duration for waveform in waveforms])

        sample_count = total_time * self.__sample_rate + 1
        ts = np.linspace(0, total_time, num=sample_count)

        channels = max([waveform.channels for waveform in waveforms])
        voltages = np.empty((len(ts), channels))
        time = 0
        for waveform in waveforms:
            indices = np.logical_and(ts >= time, ts <= time + waveform.duration)
            sample_times = ts[indices]
            offset = ts[indices][0] - time
            w_voltages = waveform.sample(sample_times, offset)
            if w_voltages.ndim == 1:
                w_voltages = w_voltages.reshape(-1,1)
            voltages[indices,:] = w_voltages
            time += waveform.duration
        return ts, voltages


def plot(pulse: SequencingElement, parameters: Dict[str, Parameter]={}, sample_rate: int=10) -> None: # pragma: no cover
    plotter = Plotter(sample_rate=sample_rate)
    sequencer = Sequencer()
    sequencer.push(pulse, parameters)
    sequence = sequencer.build()
    if not sequencer.has_finished():
        raise PlottingNotPossibleException(pulse)
    times, voltages = plotter.render(sequence)

    # plot!
    f = plt.figure()
    ax = f.add_subplot(111)
    ax.step(times, voltages, where='post')
    return f


class PlottingNotPossibleException(Exception):

    def __init__(self, pulse) -> None:
        self.pulse = pulse

    def __str__(self) -> str:
        return "Plotting is not possible. There are parameters which cannot be computed."

