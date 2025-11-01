from typing import Optional, Set, Dict, Union, Callable, Any

from qupulse import ChannelID
from qupulse.program.loop import Loop
from qupulse.program.waveforms import Waveform
from qupulse.serialization import PulseRegistryType
from qupulse.expressions import ExpressionScalar, Expression, ExpressionLike
from qupulse.parameter_scope import Scope
from qupulse.program import ProgramBuilder
from qupulse.pulses.pulse_template import PulseTemplate, AtomicPulseTemplate
from qupulse.serialization import Serializer, PulseRegistryType



class AtomicTimeReversalPulseTemplate(AtomicPulseTemplate):
    """Extend the given pulse template with a constant suffix.
    """
   
    def __init__(self, inner: PulseTemplate,
                 identifier: Optional[str] = None,
                 registry: PulseRegistryType = None):
        
        assert isinstance(inner, AtomicPulseTemplate)
        AtomicPulseTemplate.__init__(self, identifier=identifier,measurements=None)
        
        self._inner = inner        
        self._register(registry=registry)    
    
    @property
    def parameter_names(self) -> Set[str]:
        return self._inner.parameter_names
    
    @property
    def duration(self) -> ExpressionScalar:
        """An expression for the duration of this PulseTemplate."""
        return self._inner.duration
    
    @property
    def defined_channels(self) -> Set[ChannelID]:
        return self._inner.defined_channels
    
    @property
    def integral(self) -> Dict[ChannelID, ExpressionScalar]:
        return self._inner.integral

    @property
    def initial_values(self) -> Dict[ChannelID, ExpressionScalar]:
        return self._inner.final_values

    @property
    def final_values(self) -> Dict[ChannelID, ExpressionScalar]:
        return self._inner.initial_values
    
    def get_serialization_data(self, serializer: Optional[Serializer]=None) -> Dict[str, Any]:
        if serializer is not None:
            raise NotImplementedError("AtomicTimeReversalPulseTemplate does not implement legacy serialization.")
        data = super().get_serialization_data(serializer)
        data['inner'] = self._inner
        
        return data

    @classmethod
    def deserialize(cls,
                    serializer: Optional[Serializer]=None,  # compatibility to old serialization routines, deprecated
                    **kwargs) -> 'AtomicTimeReversalPulseTemplate':
        main_pt = kwargs['main_pt']
        new_duration = kwargs['new_duration']
        del kwargs['main_pt']
        del kwargs['new_duration']

        if serializer: # compatibility to old serialization routines, deprecated
            raise NotImplementedError()

        return cls(main_pt,new_duration,**kwargs)
    
    def build_waveform(self,
                       *args, **kwargs) -> Optional[Waveform]:
        wf = self._inner.build_waveform(*args, **kwargs)
        if wf is not None:
            return wf.reversed()

        

class TimeReversalPulseTemplate(PulseTemplate):
    """This pulse template reverses the inner pulse template in time."""

    def __init__(self, inner: PulseTemplate,
                 identifier: Optional[str] = None,
                 registry: PulseRegistryType = None):
        super(TimeReversalPulseTemplate, self).__init__(identifier=identifier)
        self._inner = inner
        self._register(registry=registry)

    def with_time_reversal(self) -> 'PulseTemplate':
        from qupulse.pulses import TimeReversalPT
        if self.identifier:
            return TimeReversalPT(self)
        else:
            return self._inner

    @property
    def parameter_names(self) -> Set[str]:
        return self._inner.parameter_names

    @property
    def measurement_names(self) -> Set[str]:
        return self._inner.measurement_names

    @property
    def duration(self) -> ExpressionScalar:
        return self._inner.duration

    @property
    def defined_channels(self) -> Set['ChannelID']:
        return self._inner.defined_channels

    @property
    def integral(self) -> Dict[ChannelID, ExpressionScalar]:
        return self._inner.integral
    
    @property
    def initial_values(self) -> Dict[ChannelID, ExpressionScalar]:
        return self._inner.final_values

    @property
    def final_values(self) -> Dict[ChannelID, ExpressionScalar]:
        return self._inner.initial_values
    
    def _internal_create_program(self, *, program_builder: ProgramBuilder, **kwargs) -> None:
        with program_builder.time_reversed() as reversed_builder:
            self._inner._internal_create_program(program_builder=reversed_builder, **kwargs)
            
    def build_waveform(self,
                       *args, **kwargs) -> Optional[Waveform]:
        wf = self._inner.build_waveform(*args, **kwargs)
        if wf is not None:
            return wf.reversed()

    def get_serialization_data(self, serializer=None):
        assert serializer is None, "Old stype serialization not implemented for new class"
        return {
            **super().get_serialization_data(),
            'inner': self._inner
        }

    def _is_atomic(self) -> bool:
        return self._inner._is_atomic()
    
    def pad_all_atomic_subtemplates_to(self,
        to_new_duration: Callable[[Expression], ExpressionLike]) -> 'PulseTemplate':
        self._inner = self._inner.pad_all_atomic_subtemplates_to(to_new_duration)
        return self
    