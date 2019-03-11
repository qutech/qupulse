"""This module defines plotting functionality for instantiated PulseTemplates using matplotlib.

Classes:
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
from qupulse._program.waveforms import Waveform
from qupulse._program.instructions import EXECInstruction, STOPInstruction, AbstractInstructionBlock, \
    REPJInstruction, MEASInstruction, GOTOInstruction, InstructionPointer
from qupulse._program._loop import Loop, to_waveform


__all__ = ["render", "plot", "PlottingNotPossibleException"]


def iter_waveforms(instruction_block: AbstractInstructionBlock,
                   expected_return: Optional[InstructionPointer]=None) -> Generator[Waveform, None, None]:
    # todo [2018-08-30]: seems to be unused.. remove?
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
    """Iterates over the instructions contained in an InstructionBlock (thus simulating execution).

    In effect, this function simulates the execution of the control flow represented by the passed InstructionBlock
    and returns all waveforms in the order they would be executed on the hardware, along with all measurements that
    would be made during that execution (if the extract_measurement argument is True). The waveforms are passed back
    as Waveform objects (and are not sampled at anytime during the execution of this function).

    Args:
        instruction_block: The InstructionBlock to iterate over.
        extract_measurements: If True, a list of all measurement simulated during block iteration will be returned.

    Returns:
        A tuple (waveforms, measurements, time) where waveforms is a sequence of Waveform objects in the order they
        would be executed according to the given InstructionBlock, measurements is a similar sequence of measurements
        that would be made (where each measurement is represented by a tuple (name, start_time, duration)) and time is
        the total execution duration of the block (i.e. the accumulated duration of all waveforms).
        measurements is an empty list if extract_measurements is not True.
    """
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


def render(program: Union[AbstractInstructionBlock, Loop],
           sample_rate: Real=10.0,
           render_measurements: bool=False,
           time_slice: Tuple[Real, Real]=None) -> Union[Tuple[np.ndarray, Dict[ChannelID, np.ndarray]],
                                                        Tuple[np.ndarray, Dict[ChannelID, np.ndarray],
                                                              List[MeasurementWindow]]]:
    """'Renders' a pulse program.

        Samples all contained waveforms into an array according to the control flow of the program.

        Args:
            program: The pulse (sub)program to render. Can be represented either by a Loop object or the more
                old-fashioned InstructionBlock.
            sample_rate: The sample rate in GHz.
            render_measurements: If True, the third return value is a list of measurement windows.
            time_slice: The time slice to be plotted. If None, the entire pulse will be shown.

        Returns:
            A tuple (times, values, measurements). times is a numpy.ndarray of dimensions sample_count where
            containing the time values. voltages is a dictionary of one numpy.ndarray of dimensions sample_count per
            defined channel containing corresponding sampled voltage values for that channel.
            measurements is a sequence of all measurements where each measurement is represented by a tuple
            (name, start_time, duration).
        """

    if isinstance(program, AbstractInstructionBlock):
        warnings.warn("InstructionBlock API is deprecated", DeprecationWarning)
        if time_slice is not None:
            raise ValueError("Keyword argument time_slice is not supported when rendering instruction blocks")
        return _render_instruction_block(program, sample_rate=sample_rate, render_measurements=render_measurements)
    elif isinstance(program, Loop):
        return _render_loop(program, sample_rate=sample_rate,
                            render_measurements=render_measurements, time_slice=time_slice)


def _render_instruction_block(sequence: AbstractInstructionBlock,
                              sample_rate: Real=10.0,
                              render_measurements=False) -> Union[Tuple[np.ndarray, Dict[ChannelID, np.ndarray]],
                                                                  Tuple[np.ndarray, Dict[ChannelID, np.ndarray],
                                                                        List[MeasurementWindow]]]:
    """The specific implementation of render for InstructionBlock arguments."""

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
    return times, voltages, measurements


def _render_loop(loop: Loop,
                 sample_rate: Real,
                 render_measurements: bool,
                 time_slice: Tuple[Real, Real] = None) -> Union[Tuple[np.ndarray, Dict[ChannelID, np.ndarray]],
                                                                Tuple[np.ndarray, Dict[ChannelID, np.ndarray],
                                                                List[MeasurementWindow]]]:
    """The specific implementation of render for Loop arguments."""
    waveform = to_waveform(loop)
    channels = waveform.defined_channels

    if time_slice is None:
        time_slice = (0, waveform.duration)
    elif time_slice[1] < time_slice[0] or time_slice[0] < 0 or time_slice[1] < 0:
        raise ValueError("time_slice is not valid.")

    sample_count = (time_slice[1] - time_slice[0]) * sample_rate + 1
    if sample_count<2:
        raise PlottingNotPossibleException(pulse = None, description = 'cannot render sequence with less than 2 data points')
    times = np.linspace(float(time_slice[0]), float(time_slice[1]), num=int(sample_count), dtype=float)
    times[-1] = np.nextafter(times[-1], times[-2])

    voltages = {}

    for ch in channels:
        voltages[ch] = waveform.get_sampled(ch, times)

    if render_measurements:
        measurement_dict = loop.get_measurement_windows()
        measurement_list = []
        for name, (begins, lengths) in measurement_dict.items():
            measurement_list.extend(m
                                    for m in zip(itertools.repeat(name), begins, lengths)
                                    if m[1]+m[2] > time_slice[0] and m[1] < time_slice[1])
        measurements = sorted(measurement_list, key=operator.itemgetter(1))
    else:
        measurements = []

    return times, voltages, measurements


def plot(pulse: PulseTemplate,
         parameters: Dict[str, Parameter]=None,
         sample_rate: Real=10,
         axes: Any=None,
         show: bool=True,
         plot_channels: Optional[Set[ChannelID]]=None,
         plot_measurements: Optional[Set[str]]=None,
         stepped: bool=True,
         maximum_points: int=10**6,
         time_slice: Tuple[Real, Real]=None,
         **kwargs) -> Any:  # pragma: no cover
    """Plots a pulse using matplotlib.

    The given pulse template will first be turned into a pulse program (represented by a Loop object) with the provided
    parameters. The render() function is then invoked to obtain voltage samples over the entire duration of the pulse which
    are then plotted in a matplotlib figure.

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
        time_slice: The time slice to be plotted. If None, the entire pulse will be shown.
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

    program = pulse.create_program(parameters=parameters,
                                   channel_mapping={ch: ch for ch in channels},
                                   measurement_mapping={w: w for w in pulse.measurement_names})

    if program is not None:
        times, voltages, measurements = render(program,
                                               sample_rate,
                                               render_measurements=plot_measurements,
                                               time_slice=time_slice)
    else:
        times, voltages, measurements = np.array([]), dict(), []

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

    if time_slice is None:
        time_slice = (0, duration)

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
    axes.set_xlim(-0.5+time_slice[0], time_slice[1] + 0.5)
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

    def __init__(self, pulse, description = None) -> None:
        super().__init__()
        self.pulse = pulse
        self.description = description
    def __str__(self) -> str:
        if self.description is None:
            return "Plotting is not possible. There are parameters which cannot be computed."
        else:
            return "Plotting is not possible: %s." % self.description
            

