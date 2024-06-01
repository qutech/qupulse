import numbers
from typing import Dict, Optional, Set, Union, List, Iterable

from qupulse import ChannelID
from qupulse._program.transformation import Transformation
from qupulse.parameter_scope import Scope
from qupulse.pulses.pulse_template import PulseTemplate
from qupulse.pulses import ConstantPT, SequencePT
from qupulse.expressions import ExpressionLike, ExpressionScalar
from qupulse._program.waveforms import ConstantWaveform
from qupulse.program import ProgramBuilder
from qupulse.pulses.parameters import ConstraintLike
from qupulse.pulses.measurement import MeasurementDeclaration
from qupulse.serialization import PulseRegistryType


def _evaluate_expression_dict(expression_dict: Dict[str, ExpressionScalar], scope: Scope) -> Dict[str, float]:
    return {ch: value.evaluate_in_scope(scope)
            for ch, value in expression_dict.items()}


class TimeExtensionPulseTemplate(SequencePT):
    """Extend the given pulse template with a constant(?) prefix and/or suffix.
    Both start and stop are defined as positive quantities.
    """
    
    @property
    def parameter_names(self) -> Set[str]:
        return self._extend_inner.parameter_names | set(self._extend_stop.variables) | set(self._extend_start.variables)
    
    def _create_program(self, *,
                        scope: Scope,
                        measurement_mapping: Dict[str, Optional[str]],
                        channel_mapping: Dict[ChannelID, Optional[ChannelID]],
                        global_transformation: Optional[Transformation],
                        to_single_waveform: Set[Union[str, 'PulseTemplate']],
                        program_builder: ProgramBuilder):
        
        super()._create_program(scope=scope,
                                         measurement_mapping=measurement_mapping,
                                         channel_mapping=channel_mapping,
                                         global_transformation=global_transformation,
                                         to_single_waveform=to_single_waveform | {self},
                                         program_builder=program_builder)

    def __init__(self, inner: PulseTemplate, start: ExpressionLike, stop: ExpressionLike,
                 *,
                 parameter_constraints: Optional[Iterable[ConstraintLike]]=None,
                 measurements: Optional[List[MeasurementDeclaration]]=None,
                 identifier: Optional[str] = None,
                 registry: PulseRegistryType = None
                 ):
                
        self._extend_inner = inner
        self._extend_start = ExpressionScalar(start)
        self._extend_stop = ExpressionScalar(stop)
        
        id_base = identifier if identifier is not None else ""
        
        start_pt = ConstantPT(self._extend_start,self._extend_inner.initial_values,identifier=id_base+f"__prepend_{id(self)}")
        stop_pt = ConstantPT(self._extend_stop,self._extend_inner.final_values,identifier=id_base+f"__postpend_{id(self)}")
        
        super().__init__(start_pt,self._extend_inner,stop_pt,identifier=identifier,
                       parameter_constraints=parameter_constraints,
                       measurements=measurements,
                       registry=registry)