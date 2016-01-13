import copy
from typing import Dict, List, Tuple, Set, Optional, Any, Iterable, Union
import numpy as np

"""RELATED THIRD PARTY IMPORTS"""

"""LOCAL IMPORTS"""
from qctoolkit.serialization import Serializer
from qctoolkit.expressions import Expression

from .table_pulse_template import TablePulseTemplate
from .instructions import Waveform
from .parameters import ParameterDeclaration, Parameter, ParameterNotProvidedException
from .sequence_pulse_template import MissingMappingException, MissingParameterDeclarationException, UnnecessaryMappingException

__all__ = ["MultiChannelWaveform", "MultiChannelPulseTemplate"]

Subtemplate = Tuple[PulseTemplate, Dict[str, str]]

class MultiChannelWaveform(Waveform):
    def __init__(self, waveforms: List[Waveform]):
        super().__init__()
        self.__waveforms = waveforms
        self.__channels = len(waveforms)

    @property
    def duration(self) -> float:
        return max([wf.duration for wf in self.__waveforms])

    @property
    def channels(self) -> int:
        return self.__channels

    @property
    def _compare_key(self) -> Any:
        return tuple(self.__waveforms)

    def sample(self, sample_times: numpy.ndarray, first_offset: float=0) -> np.ndarray:
        voltages = np.empty((len(sample_times), channels))
        for channel, wf in enumerate(self.__waveforms):
            voltages[:, channel] = wf.sample(sample_times, first_offset)
        return voltages

class MultiChannelPulseTemplate(SequencePulseTemplate):
    """Takes multiple TablePulseTemplates and FunctionPulseTemplates and renders them to a single waveform with multiple channels. This class works very similarly to SequencePulseTemplate. It is in fact a subclass of SequencePulseTemplate."""
    def __init__(self, subtemplates: List[Subtemplate], external_parameters: List[str], identifier: Optional[str]=None) -> None:
        # use the init function from SequencePulseTemplate.
        # That way we profit from all the consistency checks in there regarding the mappings and such.
        super().__init__(subtemplates, external_parameters, identifier)

   # def get_measurement_windows(self, parameters: Dict[str, Parameter] = None) -> List[MeasurementWindow]:
   #     raise NotImplemented()
    def build_sequence(self,
                       sequencer: Sequencer,
                       parameters: Dict[str, Parameter],
                       conditions: Dict[str, Condition],
                       instruction_block: InstructionBlock) -> None:
        missing = self.parameter_names - set(parameters)

        for m in missing:
            raise ParameterNotProvidedException(m)

        raise NotImplementedError
        #TODO: somehow join the waveforms
        # problem: the sequencer doesn't know parallel execution (yet?)

