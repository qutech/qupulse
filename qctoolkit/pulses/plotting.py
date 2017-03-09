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

from qctoolkit import ChannelID
from qctoolkit.pulses.pulse_template import PulseTemplate
from qctoolkit.pulses.parameters import Parameter
from qctoolkit.pulses.sequencing import Sequencer
from qctoolkit.pulses.instructions import EXECInstruction, STOPInstruction, InstructionSequence, \
    REPJInstruction


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

    def render(self, sequence: InstructionSequence) -> Tuple[np.ndarray, Dict[ChannelID, np.ndarray]]:
        """'Render' an instruction sequence (sample all contained waveforms into an array).

        Returns:
            a tuple (times, values) of numpy.ndarrays of similar size. times contains the time value
            of all sample times and values the corresponding sampled value.
        """
        if not all(isinstance(x, (EXECInstruction, STOPInstruction, REPJInstruction)) for x in sequence):
            raise NotImplementedError('Can only plot waveforms without branching so far.')

        def get_waveform_generator(instruction_block):
            for instruction in instruction_block:
                if isinstance(instruction, EXECInstruction):
                    yield instruction.waveform
                elif isinstance(instruction, REPJInstruction):
                    for _ in range(instruction.count):
                        yield from get_waveform_generator(instruction.target.block[instruction.target.offset:])
                else:
                    return

        waveforms = [wf for wf in get_waveform_generator(sequence)]
        if not waveforms:
            return [], []

        total_time = sum(waveform.duration for waveform in waveforms)

        channels = waveforms[0].defined_channels

        # add one sample to see the end of the waveform
        sample_count = total_time * self.__sample_rate + 1
        times = np.linspace(0, total_time, num=sample_count)
        # move the last sample inside the waveform
        times[-1] = np.nextafter(times[-1], times[-2])

        voltages = dict((ch, np.empty(len(times))) for ch in channels)
        offset = 0
        for waveform in waveforms:
            for channel in channels:
                indices = slice(*np.searchsorted(times, (offset, offset+waveform.duration)))
                sample_times = times[indices] - offset
                output_array = voltages[channel][indices]
                waveform.get_sampled(channel=channel,
                                     sample_times=sample_times,
                                     output_array=output_array)
                offset += waveform.duration
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
    channels = pulse.defined_channels

    if parameters is None:
        parameters = dict()
    plotter = Plotter(sample_rate=sample_rate)
    sequencer = Sequencer()
    sequencer.push(pulse, parameters, channel_mapping={ch: ch for ch in channels})
    sequence = sequencer.build()
    if not sequencer.has_finished():
        raise PlottingNotPossibleException(pulse)
    times, voltages = plotter.render(sequence)

    # plot to figure
    figure = plt.figure()
    ax = figure.add_subplot(111)
    for ch_name, voltage in voltages.items():
        ax.step(times, voltage, where='post', label='channel {}'.format(ch_name))

    ax.legend()

    max_voltage = max(max(channel) for channel in voltages.values())
    min_voltage = min(min(channel) for channel in voltages.values())

    # add some margins in the presentation
    plt.plot()
    plt.xlim(-0.5, times[-1] + 0.5)
    plt.ylim(min_voltage - 0.5, max_voltage + 0.5)
    plt.xlabel('Time in ns')
    plt.ylabel('Voltage')

    if pulse.identifier:
        plt.title(pulse.identifier)

    figure.show()
    return figure


class PlottingNotPossibleException(Exception):
    """Indicates that plotting is not possible because the sequencing process did not translate
    the entire given PulseTemplate structure."""

    def __init__(self, pulse) -> None:
        super().__init__()
        self.pulse = pulse

    def __str__(self) -> str:
        return "Plotting is not possible. There are parameters which cannot be computed."

