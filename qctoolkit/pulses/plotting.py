"""This module defines plotting functionality for instantiated PulseTemplates using matplotlib.

Classes:
    - Plotter: Converts an InstructionSequence into plottable time and value sample arrays.
    - PlottingNotPossibleException.
Functions:
    - plot: Plot a pulse using matplotlib.
"""

from typing import Dict, Tuple, Any, Generator, Optional

import numpy as np
import warnings

from qctoolkit.utils.types import ChannelID
from qctoolkit.pulses.pulse_template import PulseTemplate
from qctoolkit.pulses.parameters import Parameter
from qctoolkit.pulses.sequencing import Sequencer
from qctoolkit.pulses.instructions import EXECInstruction, STOPInstruction, AbstractInstructionBlock, \
    REPJInstruction, MEASInstruction, GOTOInstruction, Waveform, InstructionPointer


__all__ = ["render", "plot", "PlottingNotPossibleException"]


def iter_waveforms(instruction_block: AbstractInstructionBlock,
                   expected_return: Optional[InstructionPointer]=None) -> Generator[Waveform, None, None]:
    for i, instruction in enumerate(instruction_block):
        if isinstance(instruction, EXECInstruction):
            yield instruction.waveform
        elif isinstance(instruction, REPJInstruction):
            expected_repj_return = InstructionPointer(instruction_block, i+1)
            repj_instructions = instruction.target.block.instructions[instruction.target.offset:]
            for _ in range(instruction.count):
                yield from iter_waveforms(repj_instructions, expected_repj_return)
        elif isinstance(instruction, MEASInstruction):
            continue
        elif isinstance(instruction, GOTOInstruction):
            if instruction.target != expected_return:
                raise NotImplementedError("Instruction block contains an unexpected GOTO instruction.")
            return
        elif isinstance(instruction, STOPInstruction):
            raise StopIteration()
        else:
            raise NotImplementedError('Rendering cannot handle instructions of type {}.'.format(type(instruction)))


def render(sequence: AbstractInstructionBlock, sample_rate: float=10.0) -> Tuple[np.ndarray, Dict[ChannelID, np.ndarray]]:
    """'Render' an instruction sequence (sample all contained waveforms into an array).

    Args:
        sequence (AbstractInstructionBlock): block of instructions representing a (sub)sequence
            in the control flow of a pulse template instantiation.
        sample_rate (float): The sample rate in GHz.

    Returns:
        a tuple (times, values) of numpy.ndarrays of similar size. times contains the time value
        of all sample times and values the corresponding sampled value.
    """
    waveforms = list(iter_waveforms(sequence, ))
    if not waveforms:
        return np.empty(0), dict()

    total_time = sum(waveform.duration for waveform in waveforms)

    channels = waveforms[0].defined_channels

    # add one sample to see the end of the waveform
    sample_count = total_time * sample_rate + 1
    if not float(sample_count).is_integer():
        warnings.warn('Sample count not whole number. Casted to integer.')
    times = np.linspace(0, total_time, num=sample_count, dtype=float)
    # move the last sample inside the waveform
    times[-1] = np.nextafter(times[-1], times[-2])

    voltages = dict((ch, np.empty(len(times))) for ch in channels)
    offsets = {ch: 0 for ch in channels}
    for waveform in waveforms:
        for channel in channels:
            offset = offsets[channel]
            indices = slice(*np.searchsorted(times, (offset, offset+waveform.duration)))
            sample_times = times[indices] - offset
            output_array = voltages[channel][indices]
            waveform.get_sampled(channel=channel,
                                 sample_times=sample_times,
                                 output_array=output_array)
            offsets[channel] += waveform.duration
    return times, voltages


def plot(pulse: PulseTemplate,
         parameters: Dict[str, Parameter]=None,
         sample_rate: int=10,
         axes: Any=None,
         show: bool=True,
         plot_channels=None) -> Any:  # pragma: no cover
    """Plot a pulse using matplotlib.

    The given pulse will first be sequenced using the Sequencer class. The resulting
    InstructionSequence will be converted into sampled value arrays using the Plotter class. These
    arrays are then plotted in a matplotlib figure.

    Args:
        pulse: The pulse to be plotted.
        parameters: An optional mapping of parameter names to Parameter
            objects.
        sample_rate: The rate with which the waveforms are sampled for the plot in
            samples per time unit. (default = 10)
        axes: matplotlib Axes object the pulse will be drawn into if provided
        show: If true, the figure will be shown
    Returns:
        matplotlib.pyplot.Figure instance in which the pulse is rendered
    Raises:
        PlottingNotPossibleException if the sequencing is interrupted before it finishes, e.g.,
            because a parameter value could not be evaluated
        all Exceptions possibly raised during sequencing
    """
    from matplotlib import pyplot as plt

    channels = pulse.defined_channels

    if parameters is None:
        parameters = dict()
    sequencer = Sequencer()
    sequencer.push(pulse,
                   parameters,
                   channel_mapping={ch: ch for ch in channels},
                   window_mapping={w: w for w in pulse.measurement_names})
    sequence = sequencer.build()
    if not sequencer.has_finished():
        raise PlottingNotPossibleException(pulse)
    times, voltages = render(sequence, sample_rate)

    if axes is None:
        # plot to figure
        figure = plt.figure()
        axes = figure.add_subplot(111)
    for ch_name, voltage in voltages.items():
        if plot_channels is None or ch_name in plot_channels:
            if times.size>10e6:
                warnings.warn('plotting waveform of size %d, skipping' % times.size)
                continue
            axes.step(times, voltage, where='post', label='channel {}'.format(ch_name))

    axes.legend()

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

    if show:
        axes.get_figure().show()
    return axes.get_figure()


class PlottingNotPossibleException(Exception):
    """Indicates that plotting is not possible because the sequencing process did not translate
    the entire given PulseTemplate structure."""

    def __init__(self, pulse) -> None:
        super().__init__()
        self.pulse = pulse

    def __str__(self) -> str:
        return "Plotting is not possible. There are parameters which cannot be computed."

