"""This is the central package for defining pulses. All :class:`~qctoolkit.pulses.pulse_template.PulseTemplate`
subclasses that are final and ready to be used are imported here with their recommended abbreviation as an alias."""

from qctoolkit.pulses.function_pulse_template import FunctionPulseTemplate as FunctionPT
from qctoolkit.pulses.loop_pulse_template import ForLoopPulseTemplate as ForLoopPT
from qctoolkit.pulses.multi_channel_pulse_template import AtomicMultiChannelPulseTemplate as AtomicMultiChannelPT
from qctoolkit.pulses.pulse_template_parameter_mapping import MappingPulseTemplate as MappingPT
from qctoolkit.pulses.repetition_pulse_template import RepetitionPulseTemplate as RepetitionPT
from qctoolkit.pulses.sequence_pulse_template import SequencePulseTemplate as SequencePT
from qctoolkit.pulses.table_pulse_template import TablePulseTemplate as TablePT
from qctoolkit.pulses.point_pulse_template import PointPulseTemplate as PointPT

from qctoolkit.pulses.sequencing import Sequencer

__all__ = ["FunctionPT", "ForLoopPT", "AtomicMultiChannelPT", "MappingPT", "RepetitionPT", "SequencePT", "TablePT",
           "Sequencer", "PointPT"]

