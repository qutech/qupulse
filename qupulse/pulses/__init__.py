"""This is the central package for defining pulses. All :class:`~qupulse.pulses.pulse_template.PulseTemplate`
subclasses that are final and ready to be used are imported here with their recommended abbreviation as an alias."""

from qupulse.pulses.function_pulse_template import FunctionPulseTemplate as FunctionPT
from qupulse.pulses.loop_pulse_template import ForLoopPulseTemplate as ForLoopPT
from qupulse.pulses.multi_channel_pulse_template import AtomicMultiChannelPulseTemplate as AtomicMultiChannelPT
from qupulse.pulses.mapping_pulse_template import MappingPulseTemplate as MappingPT
from qupulse.pulses.repetition_pulse_template import RepetitionPulseTemplate as RepetitionPT
from qupulse.pulses.sequence_pulse_template import SequencePulseTemplate as SequencePT
from qupulse.pulses.table_pulse_template import TablePulseTemplate as TablePT
from qupulse.pulses.point_pulse_template import PointPulseTemplate as PointPT

import warnings
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    # ensure this is included.. it adds a deserialization handler for pulse_template_parameter_mapping.MappingPT which is not present otherwise
    import qupulse.pulses.pulse_template_parameter_mapping

__all__ = ["FunctionPT", "ForLoopPT", "AtomicMultiChannelPT", "MappingPT", "RepetitionPT", "SequencePT", "TablePT",
           "PointPT"]

