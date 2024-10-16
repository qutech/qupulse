from numbers import Real
from typing import Dict, Optional, Set, Union, List, Iterable, Any, Sequence, Hashable

from qupulse import ChannelID
from qupulse.parameter_scope import Scope
from qupulse.pulses.pulse_template import PulseTemplate, AtomicPulseTemplate
from qupulse.pulses.constant_pulse_template import ConstantPulseTemplate as ConstantPT
from qupulse.expressions import ExpressionLike, ExpressionScalar
from qupulse._program.waveforms import ConstantWaveform
from qupulse.program import ProgramBuilder
from qupulse.pulses.parameters import ConstraintLike
from qupulse.pulses.measurement import MeasurementDefiner, MeasurementDeclaration
from qupulse.serialization import Serializer, PulseRegistryType
from qupulse.program.waveforms import SequenceWaveform
from qupulse.program.transformation import Transformation, IdentityTransformation, ChainedTransformation, chain_transformations

from dataclasses import dataclass, field
from enum import Enum
from graphlib import TopologicalSorter, CycleError
from intervaltree import Interval, IntervalTree

class SpacerConstantPT(ConstantPT):
    pass

def _assemble_subset_schedule():
    pass


REF_POINT = Enum('REFERENCE_POINT', ['START', 'END',])
GAP_VOLT = Enum('GAP_VOLTAGE', ['LAST', 'NEXT', 'ZERO', 'DEFAULT'])



@dataclass(frozen=True)
class Scheduled:
    pt: PulseTemplate
    channel_subset_key: Hashable
    reference: Union['Scheduled',None]
    ref_point: REF_POINT #start,end
    rel_time: ExpressionLike
    post_gap_volt: GAP_VOLT
    pre_gap_volt: GAP_VOLT
    

@dataclass(frozen=True)
class ScheduledEvaluated(Scheduled):
    _start_time: Real 
    _duration: Real
    
    @classmethod
    def from_sched_and_scope(cls,
                             scheduled: Scheduled,
                             reference_scheduled: Union['ScheduledEvaluated',None],
                             parameters: Scope):
        
        if reference_scheduled is None:
            assert scheduled.reference is None
            assert scheduled.rel_time==0
            rel_time = 0
            start_time = 0
        else:
            rel_time = ExpressionScalar.make(scheduled.rel_time).evaluate_in_scope(parameters)
    
            if scheduled.ref_point is REF_POINT.END:
                start_time = reference_scheduled.end_time + rel_time
            elif scheduled.ref_point is REF_POINT.STRAT:
                start_time = reference_scheduled.start_time + rel_time
            else:
                raise NotImplementedError()
            
        duration = ExpressionScalar.make(scheduled.pt.duration).evaluate_in_scope(parameters)
            
        return cls(scheduled.pt,scheduled.channel_subset_key,scheduled.reference,scheduled.
                   ref_point,rel_time,scheduled.post_gap_volt,scheduled.pre_gap_volt,
                   _start_time=start_time,_duration=duration)
    
    @property
    def start_time(self,) -> Real:
        return self._start_time
    
    @property
    def duration(self,) -> Real:
        return self._duration
    
    @property
    def end_time(self,) -> Real:
        return self._start_time+self._duration


class SchedulerPT(PulseTemplate, MeasurementDefiner):
    def __init__(self,
                 channel_subsets: Dict[Hashable,Set],
                 identifier: Optional[str] = None,
                 *,
                 measurements: Optional[List[MeasurementDeclaration]]=None,
                 registry: PulseRegistryType=None) -> None:
        
         PulseTemplate.__init__(self, identifier=identifier)
         MeasurementDefiner.__init__(self, measurements=measurements)
         # AtomicPulseTemplate.__init__(self, identifier=identifier, measurements=measurements)
         
         
         
         self._channel_subsets = channel_subsets
         #no channel duplicates
         assert len(set().union(*channel_subsets.values())) == sum(len(v) for v in channel_subsets.values()),\
             'Only use disjoint subsets'
         
         # self.__pad_pt = ConstantPT(new_duration-main_pt.duration, self.final_values)
         
         # as values []
         self._scheduled = {k:set() for k in self._channel_subsets.keys()}
         
         # self._duration = ExpressionScalar.make(new_duration)
         self._duration = 0
         
         self._root: Scheduled  = None
         
         self._register(registry=registry)    
    
    
    def add_pt(self,
               pt: PulseTemplate,
               reference: None|Scheduled,
               #None: only for first
               #scheduled: on any subset
               ref_point: str = 'end',
               rel_time: ExpressionLike = 0,
               fill_post_gap_voltage: str = 'last',
               #or 'next' or 'zero'
               ) -> Scheduled:
        
        if isinstance(PulseTemplate, SchedulerPT):
            raise NotImplementedError('TBD')
        
        if self._root is not None and reference is None:
            raise NotImplementedError('always define a reference except for first addition')
   
        
        #check if correct channels
        if not sum(int(pt.defined_channels==subset) for subset in self._channel_subsets.values())==1:
            raise RuntimeError('Only define PT on exactly one subset')
        
        #find subset
        for k,subset in self._channel_subsets.items():
            if pt.defined_channels==subset:
                #add to channel subsets
                sched = Scheduled(pt,k,reference,REF_POINT[ref_point.upper()],rel_time,GAP_VOLT[fill_post_gap_voltage.upper()],
                                  GAP_VOLT.DEFAULT
                                  )
                if reference is None:
                    self._root = sched
                self._scheduled[k].add(sched)
                #return scheduled
                return sched
        
        raise RuntimeError('should not have happened')
        
        
    def build_schedule(self,
                       parameters: Scope,
                          
                       ) -> PulseTemplate:
        
        #evaluate all real timings
        pts_by_channel_subset = {k:[] for k in self._channel_subsets.keys()}
        timings_by_channel_subset = {k:IntervalTree() for k in self._channel_subsets.keys()}
        all_scheduled = set().union(*self._scheduled.values())
        scheduled_to_evaluated_scheduled = {s: None for s in all_scheduled} | {None: None}
        
        ts = TopologicalSorter()
        
        #sort for reference-resolving
        for scheduled in all_scheduled:
            if scheduled is not self._root:
                ts.add(scheduled, scheduled.reference)  # obj.pt depends on obj.reference
            else:
                ts.add(scheduled)  # root node with no dependencies
        
        try:
            sorted_scheduled = ts.static_order()
        except CycleError as e:
            print('References must not be cyclic')
            raise e
        
        #iterate through to get absolute timing
        for scheduled in sorted_scheduled:
            # scheduled = pt_to_scheduled[sorted_s.pt]
            evaluated_scheduled = ScheduledEvaluated.from_sched_and_scope(scheduled,
                                                                          scheduled_to_evaluated_scheduled[scheduled.reference],
                                                                          parameters)
            new_interval = (evaluated_scheduled.start_time, evaluated_scheduled.end_time)
            
            # Search for overlaps in the interval tree
            overlaps = timings_by_channel_subset[scheduled.channel_subset_key].overlap(*new_interval)
            if overlaps:
                raise RuntimeError(f"New scheduled PT from ({evaluated_scheduled.start_time}, {evaluated_scheduled.end_time}) overlaps with: {list(overlaps)}")
        
            # Insert the new event's interval into the tree
            timings_by_channel_subset[scheduled.channel_subset_key][new_interval[0]:new_interval[1]] = evaluated_scheduled
            
            scheduled_to_evaluated_scheduled[scheduled] = evaluated_scheduled
            
            # pts_by_channel_subset[scheduled.channel_subset_key] = evaluated_scheduled
            
        #build PT on every subset
        for k in pts_by_channel_subset.keys():
            
            #begin should be sufficient to sort by
            #returns list of (begin,end,scheduled)
            sorted_scheduled = sorted(timings_by_channel_subset[k].items(), key=lambda interval: interval.begin)
            
            first_scheduled = sorted_scheduled[0][2]
            #initial ConstantPT shouldn't hurt when length 0
            if first_scheduled.pre_gap_volt is GAP_VOLT.DEFAULT:
                pt = ConstantPT(first_scheduled.start_time, first_scheduled.pt.initial_values)
            elif first_scheduled.pre_gap_volt is GAP_VOLT.ZERO:
                pt = ConstantPT(first_scheduled.start_time, {ch: 0. for ch in self._channel_subsets.keys()})
            else:
                raise NotImplementedError()
                
            for i,interv in enumerate(sorted_scheduled[:-1]):
                s = interv[2]
                pt @= s.pt
                if sorted_scheduled[i+1][2].pre_gap_volt is not GAP_VOLT.DEFAULT:
                    raise NotImplementedError()
                if s.post_gap_volt is GAP_VOLT.NEXT:
                    gap_volt = sorted_scheduled[i+1][2].pt.initial_values
                elif s.post_gap_volt is GAP_VOLT.LAST:
                    gap_volt = s.pt.final_values
                elif s.post_gap_volt is GAP_VOLT.ZERO:
                    gap_volt = {ch: 0. for ch in self._channel_subsets.keys()}
                pt @= ConstantPT(sorted_scheduled[i+1][2].start_time-s.end_time,
                                 gap_volt
                                 )
                
            pt @= sorted_scheduled[-1][2].pt
            
            pts_by_channel_subset[k] = pt
            
        return pts_by_channel_subset
    
    @property
    def parameter_names(self) -> Set[str]:
        raise NotImplementedError()
        # return self.__main_pt.parameter_names
    
    @property
    def duration(self) -> ExpressionScalar:
        """An expression for the duration of this PulseTemplate."""
        raise NotImplementedError()

        # return self._duration
    
    @property
    def defined_channels(self) -> Set[ChannelID]:
        raise NotImplementedError()

        # return self.__main_pt.defined_channels
    
    @property
    def integral(self) -> Dict[ChannelID, ExpressionScalar]:
        raise NotImplementedError()

        
        # unextended = self.__main_pt.integral
        
        # return  {ch: unextended_ch + (self.duration-self.__main_pt.duration)*self.__main_pt.final_values[ch] \
        #          for ch,unextended_ch in unextended.items()}

    @property
    def initial_values(self) -> Dict[ChannelID, ExpressionScalar]:
        raise NotImplementedError()

        # return self.__main_pt.initial_values

    @property
    def final_values(self) -> Dict[ChannelID, ExpressionScalar]:
        raise NotImplementedError()

        # return self.__main_pt.final_values
    
    def get_serialization_data(self, serializer: Optional[Serializer]=None) -> Dict[str, Any]:
        raise NotImplementedError()

        # if serializer is not None:
        #     raise NotImplementedError("SingleWFTimeExtensionPulseTemplate does not implement legacy serialization.")
        # data = super().get_serialization_data(serializer)
        # data['main_pt'] = self.__main_pt
        # data['new_duration'] = self.duration
        # data['measurements']: self.measurement_declarations
        
        # return data

    @classmethod
    def deserialize(cls,
                    serializer: Optional[Serializer]=None,  # compatibility to old serialization routines, deprecated
                    **kwargs) -> 'SchedulerPT':
        raise NotImplementedError()

        # main_pt = kwargs['main_pt']
        # new_duration = kwargs['new_duration']
        # del kwargs['main_pt']
        # del kwargs['new_duration']

        # if serializer: # compatibility to old serialization routines, deprecated
        #     raise NotImplementedError()

        # return cls(main_pt,new_duration,**kwargs)
    
    def build_waveform(self,
                       parameters: Dict[str, Real],
                       channel_mapping: Dict[ChannelID, ChannelID]) -> SequenceWaveform:
        raise NotImplementedError()

        # return SequenceWaveform.from_sequence(
        #     [wf for sub_template in [self.__main_pt,self.__pad_pt]
        #      if (wf:=sub_template.build_waveform(parameters, channel_mapping=channel_mapping)) is not None])

    
    # def meas    

    measurement_names = MeasurementDefiner.measurement_names
    # measurement_names = None

    def _internal_create_program(self, *,
                                  scope: Scope,
                                  measurement_mapping: Dict[str, Optional[str]],
                                  channel_mapping: Dict[ChannelID, Optional[ChannelID]],
                                  global_transformation: Optional[Transformation],
                                  to_single_waveform: Set[Union[str, 'PulseTemplate']],
                                  program_builder: ProgramBuilder) -> None:
        """Parameter constraints are validated in build_waveform because build_waveform is guaranteed to be called
        during sequencing"""
        raise NotImplementedError()
