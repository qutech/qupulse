import numpy as np
from matplotlib import pyplot as plt
from itertools import chain

from .Sequencer import Sequencer, SequencingHardwareInterface, SequencingElement
from .Instructions import EXECInstruction, Waveform, WaveformTable
from .TablePulseTemplate import clean_entries, TableEntry

class PlottingDummySequencingHardware(SequencingHardwareInterface):
    def __init__(self):
        super().__init__()
        self.database = {}

    def register_waveform(self, waveform_table: WaveformTable):
        waveform = Waveform(len(waveform_table))
        wfid = hash(waveform)
        if wfid not in self.database.keys():
            self.database[wfid] = waveform_table
        return waveform


class PlottingSequencer(Sequencer):
    def __init__(self):
        self.hardware = PlottingDummySequencingHardware()
        super().__init__(self.hardware)

    def render(self):
        block = self.build()
        typecheck = lambda x: isinstance(x, EXECInstruction)
        if not all(map(typecheck, block.instructions)):
            raise NotImplementedError('Can only plot waveforms without branching so far.')
        waveforms = [self.hardware.database[hash(a.waveform)] for a in block.instructions]
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

def plot(pulse: SequencingElement, parameters={}):
    plotter = PlottingSequencer()
    if parameters:
        plotter.push(pulse, parameters)
    else:
        plotter.push(pulse)
    plotter.build()
    if not plotter.has_finished():
        raise PlottingNotPossibleException(pulse)
    times, voltages = plotter.render()
    f = plt.figure()
    ax = f.add_subplot(111)
    ax.step(times, voltages, where='post')
    f.show()

class PlottingNotPossibleException(Exception):
    def __init__(self, pulse):
        self.pulse = pulse

    def __str__(self):
        return "Plotting is not possible. {} can not be rendered for pulses that have branching.".format(self.pulse)

