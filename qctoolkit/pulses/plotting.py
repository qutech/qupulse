"""This module defines plotting functionality for instantiated PulseTemplates using matplotlib.

Classes:
    - Plotter: Converts an InstructionSequence into plottable time and value sample arrays.
    - PlottingNotPossibleException.
Functions:
    - plot: Plot a pulse using matplotlib.
"""

from typing import Dict, Tuple

import numpy as np
from matplotlib import pyplot as plt

from qctoolkit.pulses.pulse_template import PulseTemplate
from qctoolkit.pulses.parameters import Parameter
from qctoolkit.pulses.sequencing import Sequencer, SequencingElement
from qctoolkit.pulses.instructions import EXECInstruction, STOPInstruction, InstructionSequence


__all__ = ["Plotter", "plot", "PlottingNotPossibleException"]


class Plotter:
    """Plotter converts an InstructionSequence compiled by Sequencer from a PulseTemplate structure
    into a series of voltage values regularly sampled over the entire time domain for plotting.

    It currently is not able to handle instruction sequences that contain branching / jumping.
    """

    def __init__(self, sample_rate: int=10) -> None:
        """Create a new Plotter instance.

        Args:
            sample_rate (int): The sample rate in samples per time unit. (default = 10)
        """
        super().__init__()
        self.__sample_rate = sample_rate

    def render(self, sequence: InstructionSequence) -> Tuple[np.ndarray, np.ndarray]:
        """'Render' an instruction sequence (sample all contained waveforms into an array).

        Returns:
            a tuple (times, values) of numpy.ndarrays of similar size. times contains the time value
            of all sample times and values the corresponding sampled value.
        """
        if [x for x in sequence if not isinstance(x, (EXECInstruction, STOPInstruction))]:
            raise NotImplementedError('Can only plot waveforms without branching so far.')

        waveforms = [instruction.waveform
                     for instruction in sequence if isinstance(instruction, EXECInstruction)]
        if not waveforms:
            return [], []
        total_time = sum([waveform.duration for waveform in waveforms])

        sample_count = total_time * self.__sample_rate + 1
        times = np.linspace(0, total_time, num=sample_count)

        channels = max([waveform.channels for waveform in waveforms])
        voltages = np.empty((len(times), channels))
        time = 0
        for waveform in waveforms:
            indices = np.logical_and(times >= time, times <= time + waveform.duration)
            sample_times = times[indices]
            offset = times[indices][0] - time
            w_voltages = waveform.sample(sample_times, offset)
            if w_voltages.ndim == 1:
                w_voltages = w_voltages.reshape(-1,1)
            voltages[indices,:] = w_voltages
            time += waveform.duration
        return times, voltages


def plot(pulse: PulseTemplate,
         parameters: Dict[str, Parameter]=None,
         sample_rate: int=10) -> plt.Figure: # pragma: no cover
    """Plot a pulse using matplotlib.

    The given pulse will first be sequenced using the Sequencer class. The resulting
    InstructionSequence will be converted into sampled value arrays using the Plotter class. These
    arrays are then plotted in a matplotlib figure.

    Args:
        pulse (PulseTemplate): The pulse to be plotted.
        parameters (Dict(str -> Parameter)): An optional mapping of parameter names to Parameter
            objects.
        sample_rate (int): The rate with which the waveforms are sampled for the plot in
            samples per time unit. (default = 10)
    Returns:
        matplotlib.pyplot.Figure instance in which the pulse is rendered
    Raises:
        PlottingNotPossibleException if the sequencing is interrupted before it finishes, e.g.,
            because a parameter value could not be evaluated
        all Exceptions possibly raised during sequencing
    """
    if parameters is None:
        parameters = dict()
    plotter = Plotter(sample_rate=sample_rate)
    sequencer = Sequencer()
    sequencer.push(pulse, parameters)
    sequence = sequencer.build()
    if not sequencer.has_finished():
        raise PlottingNotPossibleException(pulse)
    times, voltages = plotter.render(sequence)

    # plot to figure
    figure = plt.figure()
    ax = figure.add_subplot(111)
    ax.step(times, voltages, where='post')
    return figure


class PlottingNotPossibleException(Exception):
    """Indicates that plotting is not possible because the sequencing process did not translate
    the entire given PulseTemplate structure."""

    def __init__(self, pulse) -> None:
        super().__init__()
        self.pulse = pulse

    def __str__(self) -> str:
        return "Plotting is not possible. There are parameters which cannot be computed."

