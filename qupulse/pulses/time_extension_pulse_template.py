from numbers import Real
from typing import Dict, Optional, Set, Union, List, Iterable, Any
from functools import cached_property

from qupulse import ChannelID
from qupulse.parameter_scope import Scope
from qupulse.pulses.pulse_template import PulseTemplate, AtomicPulseTemplate
from qupulse.pulses.constant_pulse_template import ConstantPulseTemplate as ConstantPT
from qupulse.expressions import ExpressionLike, ExpressionScalar
from qupulse._program.waveforms import ConstantWaveform
from qupulse.program import ProgramBuilder
from qupulse.pulses.parameters import ConstraintLike
from qupulse.pulses.measurement import MeasurementDeclaration
from qupulse.serialization import Serializer, PulseRegistryType
from qupulse.program.waveforms import SequenceWaveform


def _evaluate_expression_dict(expression_dict: Dict[str, ExpressionScalar], scope: Scope) -> Dict[str, float]:
    return {ch: value.evaluate_in_scope(scope)
            for ch, value in expression_dict.items()}


class SingleWFTimeExtensionPulseTemplate(AtomicPulseTemplate):
    """Extend the given pulse template with a constant suffix.
    """
   
    def __init__(self,
                 main_pt: PulseTemplate,
                 new_duration: Union[str, ExpressionScalar],
                 identifier: Optional[str] = None,
                 *,
                 measurements: Optional[List[MeasurementDeclaration]]=None,
                 registry: PulseRegistryType=None) -> None:
        
         AtomicPulseTemplate.__init__(self, identifier=identifier, measurements=measurements)

         self.__main_pt = main_pt
         self._duration = ExpressionScalar.make(new_duration)
         self.__pad_pt = ConstantPT(self._duration-main_pt.duration, self.final_values)

         self._register(registry=registry)    
    
    @property
    def parameter_names(self) -> Set[str]:
        return self.__main_pt.parameter_names
    
    @property
    def duration(self) -> ExpressionScalar:
        """An expression for the duration of this PulseTemplate."""
        return self._duration
    
    @property
    def defined_channels(self) -> Set[ChannelID]:
        return self.__main_pt.defined_channels
    
    @cached_property
    def integral(self) -> Dict[ChannelID, ExpressionScalar]:
        
        unextended = self.__main_pt.integral
        
        return  {ch: unextended_ch + (self.duration-self.__main_pt.duration)*self.__main_pt.final_values[ch] \
                 for ch,unextended_ch in unextended.items()}

    @property
    def initial_values(self) -> Dict[ChannelID, ExpressionScalar]:
        return self.__main_pt.initial_values

    @property
    def final_values(self) -> Dict[ChannelID, ExpressionScalar]:
        return self.__main_pt.final_values
    
    def get_serialization_data(self, serializer: Optional[Serializer]=None) -> Dict[str, Any]:
        if serializer is not None:
            raise NotImplementedError("SingleWFTimeExtensionPulseTemplate does not implement legacy serialization.")
        data = super().get_serialization_data(serializer)
        data['main_pt'] = self.__main_pt
        data['new_duration'] = self.duration
        data['measurements'] = self.measurement_declarations
        
        return data

    @classmethod
    def deserialize(cls,
                    serializer: Optional[Serializer]=None,  # compatibility to old serialization routines, deprecated
                    **kwargs) -> 'SingleWFTimeExtensionPulseTemplate':
        main_pt = kwargs['main_pt']
        new_duration = kwargs['new_duration']
        del kwargs['main_pt']
        del kwargs['new_duration']

        if serializer: # compatibility to old serialization routines, deprecated
            raise NotImplementedError()

        return cls(main_pt,new_duration,**kwargs)
    
    def build_waveform(self,
                       parameters: Dict[str, Real],
                       channel_mapping: Dict[ChannelID, ChannelID]) -> SequenceWaveform:
        return SequenceWaveform.from_sequence(
            [wf for sub_template in [self.__main_pt,self.__pad_pt]
             if (wf:=sub_template.build_waveform(parameters, channel_mapping=channel_mapping)) is not None])

    