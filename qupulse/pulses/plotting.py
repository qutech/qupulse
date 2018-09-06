"""This module defines plotting functionality for instantiated PulseTemplates using matplotlib.

Classes:
    - Plotter: Converts an InstructionSequence into plottable time and value sample arrays.
    - PlottingNotPossibleException.
Functions:
    - plot: Plot a pulse using matplotlib.
"""

from typing import Dict, Tuple, Any, Generator, Optional, Set, List, Union
from numbers import Real

import numpy as np
import warnings
import operator
import itertools

from qupulse.utils.types import ChannelID, MeasurementWindow
from qupulse.pulses.pulse_template import PulseTemplate
from qupulse.pulses.parameters import Parameter
from qupulse.pulses.sequencing import Sequencer
from qupulse._program.waveforms import Waveform
from qupulse._program.instructions import EXECInstruction, STOPInstruction, AbstractInstructionBlock, \
    REPJInstruction, MEASInstruction, GOTOInstruction, InstructionPointer
from qupulse._program._loop import Loop, to_waveform


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
            return
        else:
            raise NotImplementedError('Rendering cannot handle instructions of type {}.'.format(type(instruction)))


def iter_instruction_block(instruction_block: AbstractInstructionBlock,
                           extract_measurements: bool) -> Tuple[list, list, Real]:
    block_stack = [(enumerate(instruction_block), None)]
    waveforms = []
    measurements = []
    time = 0

    while block_stack:
        block, expected_return = block_stack.pop()

        for i, instruction in block:
            if isinstance(instruction, EXECInstruction):
                waveforms.append(instruction.waveform)
                time += instruction.waveform.duration
            elif isinstance(instruction, REPJInstruction):
                expected_repj_return = InstructionPointer(instruction_block, i+1)
                repj_instructions = instruction.target.block.instructions[instruction.target.offset:]

                block_stack.append((block, expected_return))
                block_stack.extend((enumerate(repj_instructions), expected_repj_return)
                                   for _ in range(instruction.count))
                break
            elif isinstance(instruction, MEASInstruction):
                if extract_measurements:
                    measurements.extend((name, begin+time, length)
                                        for name, begin, length in instruction.measurements)
            elif isinstance(instruction, GOTOInstruction):
                if instruction.target != expected_return:
                    raise NotImplementedError("Instruction block contains an unexpected GOTO instruction.")
                break
            elif isinstance(instruction, STOPInstruction):
                block_stack.clear()
                break
            else:
                raise NotImplementedError('Rendering cannot handle instructions of type {}.'.format(type(instruction)))

    return waveforms, measurements, time


def render(sequence: Union[AbstractInstructionBlock, Loop],
           sample_rate: Real=10.0,
           render_measurements=False) -> Union[Tuple[np.ndarray, Dict[ChannelID, np.ndarray]],
                                               Tuple[np.ndarray, Dict[ChannelID, np.ndarray], List[MeasurementWindow]]]:
    """'Render' an instruction sequence (sample all contained waveforms into an array).
        TODO: update docstring
        Args:
            sequence (AbstractInstructionBlock): block of instructions representing a (sub)sequence
                in the control flow of a pulse template instantiation.
            sample_rate (float): The sample rate in GHz.
            render_measurements: If True, the third return value is a list of measurement windows

        Returns:
            a tuple (times, values) of numpy.ndarrays of similar size. times contains the time value
            of all sample times and values the corresponding sampled value.
        """

    if isinstance(sequence, AbstractInstructionBlock):
        return _render_instruction_block(sequence, sample_rate=sample_rate, render_measurements=render_measurements)
    elif isinstance(sequence, Loop):
        return _render_loop(sequence, sample_rate=sample_rate, render_measurements=render_measurements)


def _render_instruction_block(sequence: AbstractInstructionBlock,
                              sample_rate: Real=10.0,
                              render_measurements=False) -> Union[Tuple[np.ndarray, Dict[ChannelID, np.ndarray]],
                                                                  Tuple[np.ndarray, Dict[ChannelID, np.ndarray],
                                                                        List[MeasurementWindow]]]:

    waveforms, measurements, total_time = iter_instruction_block(sequence, render_measurements)
    if not waveforms:
        return np.empty(0), dict()

    channels = waveforms[0].defined_channels

    # add one sample to see the end of the waveform
    sample_count = total_time * sample_rate + 1
    if not float(sample_count).is_integer():
        warnings.warn('Sample count not whole number. Casted to integer.')
    times = np.linspace(0, float(total_time), num=int(sample_count), dtype=float)
    # move the last sample inside the waveform
    times[-1] = np.nextafter(times[-1], times[-2])

    voltages = dict((ch, np.empty(len(times))) for ch in channels)
    offset = 0
    for waveform in waveforms:
        wf_end = offset + waveform.duration
        indices = slice(*np.searchsorted(times, (offset, wf_end)))
        sample_times = times[indices] - float(offset)
        for channel in channels:
            output_array = voltages[channel][indices]
            waveform.get_sampled(channel=channel,
                                 sample_times=sample_times,
                                 output_array=output_array)
        assert(output_array.shape == sample_times.shape)
        offset = wf_end
    if render_measurements:
        return times, voltages, measurements
    else:
        return times, voltages


def _render_loop(loop: Loop,
                 sample_rate: Real,
                 render_measurements: bool) -> Union[Tuple[np.ndarray, Dict[ChannelID, np.ndarray]],
                                                     Tuple[np.ndarray, Dict[ChannelID, np.ndarray],
                                                           List[MeasurementWindow]]]:
    waveform = to_waveform(loop)
    channels = waveform.defined_channels

    sample_count = waveform.duration * sample_rate + 1
    times = np.linspace(0., float(waveform.duration), num=int(sample_count), dtype=float)
    times[-1] = np.nextafter(times[-1], times[-2])

    voltages = {}

    for ch in channels:
        voltages[ch] = waveform.get_sampled(ch, times)

    if render_measurements:
        measurement_dict = loop.get_measurement_windows()
        measurement_list = []
        for name, (begins, lengths) in measurement_dict.items():
            measurement_list.extend(zip(itertools.repeat(name), begins, lengths))
        measurements = sorted(measurement_list, key=operator.itemgetter(1))

        return times, voltages, measurements
    else:
        return times, voltages


def plot(pulse: PulseTemplate,
         parameters: Dict[str, Parameter]=None,
         sample_rate: Real=10,
         axes: Any=None,
         show: bool=True,
         plot_channels: Optional[Set[ChannelID]]=None,
         plot_measurements: Optional[Set[str]]=None,
         stepped: bool=True,
         maximum_points: int=10**6,
         **kwargs) -> Any:  # pragma: no cover
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
        plot_channels: If specified only channels from this set will be plotted. If omitted all channels will be.
        stepped: If true pyplot.step is used for plotting
        plot_measurements: If specified measurements in this set will be plotted. If omitted no measurements will be.
        maximum_points: If the sampled waveform is bigger, it is not plotted
        kwargs: Forwarded to pyplot. Overwrites other settings.
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

    if plot_measurements:
        times, voltages, measurements = render(sequence, sample_rate, render_measurements=True)
    else:
        times, voltages = render(sequence, sample_rate)

    duration = 0
    if times.size == 0:
        warnings.warn("Pulse to be plotted is empty!")
    elif times.size > maximum_points:
        # todo [2018-05-30]: since it results in an empty return value this should arguably be an exception, not just a warning
        warnings.warn("Sampled pulse of size {wf_len} is lager than {max_points}".format(wf_len=times.size,
                                                                                         max_points=maximum_points))
        return None
    else:
        duration = times[-1]

    legend_handles = []
    if axes is None:
        # plot to figure
        figure = plt.figure()
        axes = figure.add_subplot(111)

    if plot_channels is not None:
        voltages = {ch: voltage
                    for ch, voltage in voltages.items()
                    if ch in plot_channels}

    for ch_name, voltage in voltages.items():
        label = 'channel {}'.format(ch_name)
        if stepped:
            line, = axes.step(times, voltage, **{**dict(where='post', label=label), **kwargs})
        else:
            line, = axes.plot(times, voltage, **{**dict(label=label), **kwargs})
        legend_handles.append(line)

    if plot_measurements:
        measurement_dict = dict()
        for name, begin, length in measurements:
            if name in plot_measurements:
                measurement_dict.setdefault(name, []).append((begin, begin+length))

        color_map = plt.cm.get_cmap('plasma')
        meas_colors = {name: color_map(i/len(measurement_dict))
                       for i, name in enumerate(measurement_dict.keys())}
        for name, begin_end_list in measurement_dict.items():
            for begin, end in begin_end_list:
                poly = axes.axvspan(begin, end, alpha=0.2, label=name, edgecolor='black', facecolor=meas_colors[name])
            legend_handles.append(poly)

    axes.legend(handles=legend_handles)

    max_voltage = max((max(channel, default=0) for channel in voltages.values()), default=0)
    min_voltage = min((min(channel, default=0) for channel in voltages.values()), default=0)

    # add some margins in the presentation
    axes.set_xlim(-0.5, duration + 0.5)
    axes.set_ylim(min_voltage - 0.1*(max_voltage-min_voltage), max_voltage + 0.1*(max_voltage-min_voltage))
    axes.set_xlabel('Time (ns)')
    axes.set_ylabel('Voltage (a.u.)')

    if pulse.identifier:
        axes.set_title(pulse.identifier)

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

