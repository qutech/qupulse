import numpy as np
from matplotlib import pyplot as plt
from typing import Dict

from .Parameter import Parameter
from .Sequencer import Sequencer, SequencingHardwareInterface, SequencingElement
from .Instructions import EXECInstruction, Waveform, WaveformTable, InstructionBlock
from .TablePulseTemplate import clean_entries, TableEntry


class Plotter(SequencingHardwareInterface):

    def __init__(self) -> None:
        super().__init__()
        self.__database = {}

    def register_waveform(self, waveform_table: WaveformTable) -> None:
        waveform = Waveform(len(waveform_table))
        waveform_id = hash(waveform)
        if waveform_id not in self.__database.keys():
            self.__database[waveform_id] = waveform_table
        return waveform

    def render(self, block: InstructionBlock) -> None:
        if not all(map(lambda x: isinstance(x, EXECInstruction), block.instructions)):
            raise NotImplementedError('Can only plot waveforms without branching so far.')
        waveforms = [self.__database[hash(a.waveform)] for a in block.instructions]
        total_time = 0
        total_waveform = [waveforms[0][0]]
        for wf in waveforms:
            for point in wf:
                new_time = point.t + total_time
                if new_time != total_time:
                    total_waveform.append(TableEntry(new_time, point.v, point.interp))
            total_time += wf[-1].t
        entries = clean_entries(total_waveform)
        ts = np.arange(0, entries[-1].t, 1)

        voltages = np.empty_like(ts) # prepare voltage vector
        for entry1, entry2 in zip(entries[:-1], entries[1:]): # iterate over interpolated areas
            indices = np.logical_and(ts >= entry1.t, ts <= entry2.t)
            voltages[indices] = entry2.interp(entry1, entry2, ts[indices]) # evaluate interpolation at each time
        return ts, voltages


def plot(pulse: SequencingElement, parameters: Dict[str, Parameter]={}) -> None:
    plotter = Plotter()
    sequencer = Sequencer(plotter)
    if parameters:
        sequencer.push(pulse, parameters)
    else:
        sequencer.push(pulse)
    block = sequencer.build()
    if not sequencer.has_finished():
        raise PlottingNotPossibleException(pulse)
    times, voltages = plotter.render(block)
    f = plt.figure()
    ax = f.add_subplot(111)
    ax.step(times, voltages, where='post')
    f.show()


class PlottingNotPossibleException(Exception):
    def __init__(self, pulse) -> None:
        self.pulse = pulse

    def __str__(self) -> str:
        return "Plotting is not possible. {} can not be rendered for pulses that have branching.".format(self.pulse)

