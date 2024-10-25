from numbers import Real
from typing import Dict, Optional, Set, Union, List, Iterable, Any, Sequence, Hashable, Mapping, Generator, Tuple, Self

from qupulse import ChannelID
from qupulse.parameter_scope import Scope
from qupulse.pulses.pulse_template import PulseTemplate, AtomicPulseTemplate
from qupulse.pulses.constant_pulse_template import ConstantPulseTemplate as ConstantPT
# from qupulse.pulses.point_pulse_template import PointPulseTemplate as PointPT
from qupulse.pulses.table_pulse_template import TablePulseTemplate as TablePT

from qupulse.expressions import ExpressionLike, ExpressionScalar, Expression
from qupulse.expressions.simple import SimpleExpression
from qupulse._program.waveforms import ConstantWaveform
from qupulse.program import ProgramBuilder, Program
from qupulse.program.linspace import LinSpaceBuilder
from qupulse.program.multi import MultiProgramBuilder
from qupulse.utils import cached_property, flatten_dict_to_sets

from qupulse.pulses.parameters import ConstraintLike
from qupulse.pulses.measurement import MeasurementDefiner, MeasurementDeclaration
from qupulse.serialization import Serializer, PulseRegistryType
from qupulse.program.waveforms import SequenceWaveform
from qupulse.program.transformation import Transformation, IdentityTransformation, ChainedTransformation, chain_transformations

from dataclasses import dataclass, field
from enum import Enum
from graphlib import TopologicalSorter, CycleError
from intervaltree import Interval, IntervalTree
from sympy import Max as spMax
import sympy as sp
from copy import deepcopy

import numpy as np

class SpacerConstantPT(ConstantPT):
    pass

def _assemble_subset_schedule():
    pass


REF_POINT = Enum('REFERENCE_POINT', ['START', 'END',])
GAP_VOLT = Enum('GAP_VOLTAGE', ['LAST', 'NEXT', 'ZERO', 'DEFAULT'])
#define default as last if there is last, otherwise next

SubsetID = str

@dataclass(frozen=True)
class Scheduled:
    pt: PulseTemplate
    channel_subset_key: Hashable
    reference: Union['Scheduled',None]
    ref_point: REF_POINT #start,end
    rel_time: ExpressionScalar
    post_gap_volt: GAP_VOLT
    pre_gap_volt: GAP_VOLT
    
    def __str__(self) -> str:
        # return "("+",".join(str(self.pt),)+")"
        return str(self.pt)

    

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
            elif scheduled.ref_point is REF_POINT.START:
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


class SchedulerPulseTemplate(PulseTemplate, MeasurementDefiner):
    
    CONSTANT_TIME_THRESHOLD = 128
    
    def __init__(self,
                 channel_subsets: Dict[Hashable,Set],
                 identifier: Optional[str] = None,
                 *,
                 empty_fill: GAP_VOLT = GAP_VOLT.DEFAULT,
                 measurements: Optional[List[MeasurementDeclaration]]=None,
                 registry: PulseRegistryType=None) -> None:
        
         PulseTemplate.__init__(self, identifier=identifier)
         MeasurementDefiner.__init__(self, measurements=measurements)
         # AtomicPulseTemplate.__init__(self, identifier=identifier, measurements=measurements)
         

         self._channel_subsets = channel_subsets
         #no channel duplicates
         assert len(set().union(*channel_subsets.values())) == sum(len(v) for v in channel_subsets.values()),\
             'Only use disjoint subsets'
         
         self._scheduled: Dict[str,Set[Scheduled]] = {k:set() for k in self._channel_subsets.keys()}
         self._root: Scheduled  = None
         
         self._register(registry=registry)    
    
         self._incoming_volts: Dict[ChannelID, float|SimpleExpression] = {ch: None for ch in self.defined_channels}
        
         self._empty_fill = empty_fill
        
    def _set_incoming_volts(self, volt_dict: Dict[ChannelID, float|SimpleExpression]):
        
        assert set(volt_dict.keys()) == set(self._incoming_volts.keys())
        self._incoming_volts = deepcopy(volt_dict)
        
    def _get_incoming_volts(self) -> Optional[Dict[ChannelID, float|SimpleExpression]]:
        
        # assert not any(v is None for v in self._incoming_volts.values()), 'Undefined incoming volt'
        if any(v is None for v in self._incoming_volts.values()): return None
        return self._incoming_volts
        
    
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
        
        
        self.__dict__.pop('parameter_names', None)
        self.__dict__.pop('duration', None)
        self.__dict__.pop('sorted_scheduled', None)
        self.__dict__.pop('_start_points_by_subset', None)
        self.__dict__.pop('initial_values', None)
        self.__dict__.pop('final_values', None)
        self.__dict__.pop('defined_channels', None)

        
        
        
        if self._root is not None and reference is None:
            raise NotImplementedError('always define a reference except for first addition')
   
        
        # if isinstance(PulseTemplate, SchedulerPulseTemplate):
        #     #check if correct channels
        #     if not sum(int(pt.defined_subsets==subset) for subset in self._channel_subsets.values())==1:
        #         raise RuntimeError('Only define PT on exactly one subset')
        # else:
        #check if correct channels
        if not sum(int(pt.defined_channels==subset) for subset in self._flattened_channels_by_subset_key.values())==1:
            print(pt.defined_channels)
            print(self._flattened_channels_by_subset_key.values())
            raise RuntimeError('Only define PT on exactly one subset')
        
        #find subset
        # for k,subset in self._channel_subsets.items():
        for k,subset in self._flattened_channels_by_subset_key.items():

            if pt.defined_channels==subset:
                #add to channel subsets
                sched = Scheduled(pt,k,reference,REF_POINT[ref_point.upper()],ExpressionScalar(rel_time),GAP_VOLT[fill_post_gap_voltage.upper()],
                                  GAP_VOLT.DEFAULT
                                  )
                if reference is None:
                    self._root = sched
                self._scheduled[k].add(sched)
                #return scheduled
                return sched
        
        raise RuntimeError('should not have happened')
        
    @property
    def sorted_scheduled(self) -> Generator[Scheduled,None,None]:
        
        all_scheduled = set().union(*self._scheduled.values())

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
            
        return sorted_scheduled
    
    # def _recursive_build_schedule
    
    def build_schedule(self,
                       parameters: Scope,
                       scheduler_options: Dict[str,Any],
                       ) -> Dict[str,PulseTemplate]:
                       # ) -> Dict[str,PulseTemplate|Self]:
        
        constant_as_arbitrary_wf_below = scheduler_options.get('constant_val_time_threshold',self.CONSTANT_TIME_THRESHOLD)                   
        
        #evaluate all real timings
        pts_by_channel_subset = {k:None for k in self._channel_subsets.keys()}
        # pts_by_channel_subset = deepcopy(self._channel_subsets)

        timings_by_channel_subset = {k:IntervalTree() for k in self._channel_subsets.keys()}
        all_scheduled = set().union(*self._scheduled.values())
        scheduled_to_evaluated_scheduled = {s: None for s in all_scheduled} | {None: None}
        
        sorted_scheduled = self.sorted_scheduled
        
        offset_start_time = 0.
        end_time = -np.inf
        
        #iterate through to get absolute timing
        for scheduled in sorted_scheduled:
            # scheduled = pt_to_scheduled[sorted_s.pt]
            evaluated_scheduled = ScheduledEvaluated.from_sched_and_scope(scheduled,
                                                                          scheduled_to_evaluated_scheduled[scheduled.reference],
                                                                          parameters)
            new_interval = (evaluated_scheduled.start_time, evaluated_scheduled.end_time)
            
            if evaluated_scheduled.start_time < offset_start_time:
                offset_start_time = evaluated_scheduled.start_time
                
            if evaluated_scheduled.end_time > end_time:
                end_time = evaluated_scheduled.end_time
            
            # Search for overlaps in the interval tree
            overlaps = timings_by_channel_subset[scheduled.channel_subset_key].overlap(*new_interval)
            if overlaps:
                raise RuntimeError(f"New scheduled PT from ({evaluated_scheduled.start_time}, {evaluated_scheduled.end_time}) overlaps with: {list(overlaps)}")
        
            # Insert the new event's interval into the tree
            timings_by_channel_subset[scheduled.channel_subset_key][new_interval[0]:new_interval[1]] = evaluated_scheduled
            
            scheduled_to_evaluated_scheduled[scheduled] = evaluated_scheduled
            
            # pts_by_channel_subset[scheduled.channel_subset_key] = evaluated_scheduled
        
        #!!! add lowest start time as offset, latest time as end time
        
        #build PT on every subset
        def nested_pt_subsets(pts_by_channel_subset):
            for k in pts_by_channel_subset.keys():
                # if isinstance(pts_by_channel_subset[k],dict):
                #     return nested_pt_subsets(pts_by_channel_subset[k])
                # assert isinstance(pts_by_channel_subset[k],Set)
                
                #begin should be sufficient to sort by
                #returns list of (begin,end,scheduled)
                sorted_scheduled = sorted(timings_by_channel_subset[k].items(), key=lambda interval: interval.begin)
                
                if len(sorted_scheduled)==0:
                    # raise NotImplementedError('one pt on every subset')
                    if self._empty_fill is GAP_VOLT.DEFAULT:
                        if (potential_previous_volts:=self._get_incoming_volts()) is not None:
                            empty_fill = {ch: potential_previous_volts[ch] for ch in self._flattened_channels_by_subset_key[k]}
                        else:
                            empty_fill = {ch:0. for ch in self._flattened_channels_by_subset_key[k]}
                        pt = ConstantPT(end_time-offset_start_time,empty_fill)
                        
                    else:
                        raise NotImplementedError()
                        
                    pts_by_channel_subset[k] = pt
                        
                    continue
                
                first_scheduled = sorted_scheduled[0][2]
                
                #!!! optional short fillers as timeextendpt / non-constant pt as  hdawg bad with short hold times?
                
                #initial ConstantPT shouldn't hurt when length 0
                #... but maybe do when finite but too short
                
                #!!! if sub.pt is SchedulerPulseTemplate, add incoming volts
                
                if first_scheduled.pre_gap_volt is GAP_VOLT.DEFAULT:
                    if (potential_previous_volts:=self._get_incoming_volts()) is not None:
                        first_gap_volt = {ch: potential_previous_volts[ch] for ch in first_scheduled.pt.defined_channels}
                    else:
                        first_gap_volt = first_scheduled.pt.initial_values
                elif first_scheduled.pre_gap_volt is GAP_VOLT.ZERO:
                    first_gap_volt = {ch:0. for ch in self._flattened_channels_by_subset_key[k]}
                elif first_scheduled.pre_gap_volt is GAP_VOLT.NEXT:
                    first_gap_volt = first_scheduled.pt.initial_values
                elif first_scheduled.pre_gap_volt is GAP_VOLT.LAST:
                    potential_previous_volts = self._get_incoming_volts()
                    assert potential_previous_volts is not None, 'Undefined incoming volt'
                    first_gap_volt = {ch: potential_previous_volts[ch] for ch in first_scheduled.pt.defined_channels}
                    raise NotImplementedError()
                
                if first_scheduled.start_time-offset_start_time<constant_as_arbitrary_wf_below:
                    #uglily set two table entries for now to force non-constant waveform generation
                    pt = TablePT({ch: [(0.,0.),
                                       (first_scheduled.start_time-offset_start_time,val,'jump')] for ch,val in first_gap_volt.items()},
                                 allow_constant_waveform=False)
                else:
                    pt = ConstantPT(first_scheduled.start_time-offset_start_time, first_gap_volt)
            
                if isinstance(first_scheduled,SchedulerPulseTemplate):
                    first_scheduled._set_incoming_volts(first_gap_volt)
                
                
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
                        gap_volt = {ch: 0. for ch in self._flattened_channels_by_subset_key[k]}
                        
                    if sorted_scheduled[i+1][2].start_time-s.end_time<constant_as_arbitrary_wf_below:
                        #uglily set two table entries for now to force non-constant waveform generation
                        pt @= TablePT({ch: [(0.,0.),
                                            (sorted_scheduled[i+1][2].start_time-s.end_time,val,'jump')] for ch,val in gap_volt.items()},
                                      allow_constant_waveform=False)
                    else:
                        pt @= ConstantPT(sorted_scheduled[i+1][2].start_time-s.end_time,
                                         gap_volt
                                         )
                        
                    if isinstance(sorted_scheduled[i+1][2],SchedulerPulseTemplate):
                        sorted_scheduled[i+1][2]._set_incoming_volts(gap_volt)
                    
                s = sorted_scheduled[-1][2]
                pt @= s.pt
                
                #!!! gap after last
                if s.post_gap_volt is GAP_VOLT.NEXT:
                    #can probably be done with volatile params, but may be unnecessary anyway...
                    raise NotImplementedError()
                elif s.post_gap_volt is GAP_VOLT.LAST:
                    gap_volt = s.pt.final_values
                elif s.post_gap_volt is GAP_VOLT.ZERO:
                    gap_volt = {ch: 0. for ch in self._flattened_channels_by_subset_key.values()}
                if end_time-offset_start_time-sorted_scheduled[-1][2].end_time<constant_as_arbitrary_wf_below:
                    #uglily set two table entries for now to force non-constant waveform generation
                    pt @= TablePT({ch: [(0.,0.),
                                        (end_time-offset_start_time-sorted_scheduled[-1][2].end_time,val,'jump')] for ch,val in gap_volt.items()},
                                  allow_constant_waveform=False)
                else:
                    pt @= ConstantPT(end_time-offset_start_time-sorted_scheduled[-1][2].end_time,
                                     gap_volt
                                     )
                
                
                pts_by_channel_subset[k] = pt
        
        nested_pt_subsets(pts_by_channel_subset)
        
        return pts_by_channel_subset
    
    @cached_property
    def parameter_names(self) -> Set[str]:
        parameter_names = set()
        for subset_scheduled in self._scheduled.values():
            for sched in subset_scheduled:
                parameter_names = parameter_names.union(sched.pt.parameter_names)
        return parameter_names
    
    @cached_property
    def _start_points_by_subset(self) -> Dict[Scheduled,ExpressionScalar]:
        sorted_scheduled = self.sorted_scheduled
        initial_scheduled = next(sorted_scheduled)
        #pt.duration should always return ExpressionScalar
        # duration = initial_scheduled.pt.duration
        # duration = ExpressionScalar(0.)
        start_points_by_scheduled = {initial_scheduled: ExpressionScalar(0.)}
        
        # sympy_max = ExpressionScalar(f'Max({a.underlying_expression},b.underlying_expression)')
        
        #remaining:
        for scheduled in sorted_scheduled:
            start_points_by_scheduled[scheduled] = start_points_by_scheduled[scheduled.reference] + scheduled.rel_time

            if scheduled.ref_point is REF_POINT.END:
                start_points_by_scheduled[scheduled] += scheduled.reference.pt.duration
                
            elif scheduled.ref_point is REF_POINT.START:
                pass
            else:
                raise NotImplementedError()
        return start_points_by_scheduled
                
    @cached_property
    def duration(self) -> ExpressionScalar:
        # this is unsafe as no check for timing consistency is made
        
        start_points_by_scheduled = self._start_points_by_subset
        
        max_time = ExpressionScalar(spMax(*[start+s.pt.duration for s,start in start_points_by_scheduled.items()]))
        
        return max_time
        
    @cached_property
    def _flattened_channels_by_subset_key(self) -> Dict[SubsetID,Set[ChannelID]]:
        return flatten_dict_to_sets(self._channel_subsets)
    
    @cached_property
    def defined_channels(self) -> Set[ChannelID]:
        
        # flattened_channels = flatten_dict_to_sets(self._channel_subsets)
        
        return set().union(*[chs for chs in self._flattened_channels_by_subset_key.values()])
        # return set().union(*[subset for subset in self._channel_subsets.keys()])

        # return set().union(*[chs  for s in self._scheduled.values() for chs in s.pt.defined_channels])
    
    # @property
    # def defined_channel_ids(self) -> Set[SubsetID]:
    #     # return set().union(*[chs for chs in self._channel_subsets.values()])
    #     return set().union(*[chs  for s in self._scheduled.values() for chs in s.pt.defined_channels])
    @property
    def defined_subsets(self) -> Set[SubsetID]:
        # return set().union(*[chs for chs in self._channel_subsets.keys()])
        return set(self._channel_subsets.keys())

        # return set().union(*[chs  for s in self._scheduled.values() for chs in s.pt.defined_channels])


    @property
    def integral(self) -> Dict[ChannelID, ExpressionScalar]:
        raise NotImplementedError()



    @cached_property
    def initial_values(self) -> Dict[ChannelID, ExpressionScalar]:
        
        start_points_by_scheduled = self._start_points_by_subset
        
        initial_values = {}
        
        for key,ch_subset in self._channel_subsets.items():
            subset_scheduled = {s:time for s,time in start_points_by_scheduled.items() if s.pt.defined_channels==ch_subset}
            initial_values.update(get_symbolic_vals_with_conditions_from_dict(subset_scheduled,
                                                                              # ch_subset=self._channel_subsets[key],
                                                                              ch_subset=self._flattened_channels_by_subset_key[key],
                                                                              min_val=True))
        
        return initial_values
        

    @cached_property
    def final_values(self) -> Dict[ChannelID, ExpressionScalar]:
        
        start_points_by_scheduled = self._start_points_by_subset
        
        final_values = {}
        
        for key,ch_subset in self._channel_subsets.items():
            subset_scheduled = {s:time for s,time in start_points_by_scheduled.items() if s.pt.defined_channels==ch_subset}
            final_values.update(get_symbolic_vals_with_conditions_from_dict(subset_scheduled,
                                                                            # ch_subset=self._channel_subsets[key],
                                                                            ch_subset=self._flattened_channels_by_subset_key[key],
                                                                            min_val=False))
            
        return final_values
    
    def get_serialization_data(self, serializer: Optional[Serializer]=None) -> Dict[str, Any]:

        if serializer is not None:
            raise NotImplementedError("SchedulerPulseTemplate does not implement legacy serialization.")
        
        data = super().get_serialization_data(serializer)

        data['channel_subsets'] = self._channel_subsets
        data['scheduled'] = self._scheduled
        data['root'] = self._root
                

        if self.measurement_declarations:
            raise NotImplementedError()
            data['measurements'] = self.measurement_declarations

        return data
        
    @classmethod
    def deserialize(cls,
                    serializer: Optional[Serializer]=None,  # compatibility to old serialization routines, deprecated
                    **kwargs) -> 'SchedulerPulseTemplate':
        raise NotImplementedError()

        # main_pt = kwargs['main_pt']
        # new_duration = kwargs['new_duration']
        # del kwargs['main_pt']
        # del kwargs['new_duration']

        # if serializer: # compatibility to old serialization routines, deprecated
        #     raise NotImplementedError()

        # return cls(main_pt,new_duration,**kwargs)
    
    def __str__(self) -> str:
        st = "SchedulerPT: "
        for k, scheduled_set in self._scheduled.items():
            st += f"subset {k}: "
            for s in scheduled_set:
                st += str(s)+";"
            
        return +";".join(self._sch, self.body)
    
    
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
    
    
    
    #need to overwrite?
    def _create_program(self, *,
                        scope: Mapping[str,Scope],
                        measurement_mapping: Dict[str, Optional[str]],
                        channel_mapping: Dict[ChannelID, Optional[ChannelID]],
                        global_transformation: Optional[Transformation],
                        to_single_waveform: Set[Union[str, 'PulseTemplate']],
                        program_builder: ProgramBuilder):
        """Generic part of create program. This method handles to_single_waveform and the configuration of the
        transformer."""
        if self.identifier in to_single_waveform or self in to_single_waveform:
            raise NotImplementedError()
            # with program_builder.new_subprogram(global_transformation=global_transformation) as inner_program_builder:

            #     if not scope.get_volatile_parameters().keys().isdisjoint(self.parameter_names):
            #         raise NotImplementedError('A pulse template that has volatile parameters cannot be transformed into a '
            #                                   'single waveform yet.')

            #     self._internal_create_program(scope=scope,
            #                                   measurement_mapping=measurement_mapping,
            #                                   channel_mapping=channel_mapping,
            #                                   global_transformation=None,
            #                                   to_single_waveform=to_single_waveform,
            #                                   program_builder=inner_program_builder)

        else:
            self._internal_create_program(scope=scope,
                                          measurement_mapping=measurement_mapping,
                                          channel_mapping=channel_mapping,
                                          to_single_waveform=to_single_waveform,
                                          global_transformation=global_transformation,
                                          program_builder=program_builder)
    
    
    def _internal_create_program(self, *,
                                  # scope: Scope|Dict[str,Scope],
                                  scope: Scope|Tuple[Scope,'ForLoopPT'],

                                  measurement_mapping: Dict[str, Optional[str]],
                                  channel_mapping: Dict[ChannelID, Optional[ChannelID]],
                                  global_transformation: Optional[Transformation],
                                  to_single_waveform: Set[Union[str, 'PulseTemplate']],
                                  program_builder: ProgramBuilder) -> None:
        """Parameter constraints are validated in build_waveform because build_waveform is guaranteed to be called
        during sequencing"""
        # raise NotImplementedError()
        
        if not all(k==v or v is None for k,v in channel_mapping.items()):
            raise NotImplementedError()
        
        if not isinstance(program_builder,MultiProgramBuilder):
            raise NotImplementedError()

        
        if (mode:=program_builder._stack[-1][0]) in {"top","sequence"}:
            # pt_dict = self.build_schedule({key:scope for key in self._channel_subsets.keys()})
            pt_dict = self.build_schedule(scope,program_builder._scheduler_options)

            for key,subset_program_builder in program_builder._stack[-1][1].items():
                pt_dict[key]._create_program(scope=scope,
                                             measurement_mapping=measurement_mapping,
                                             channel_mapping=channel_mapping,
                                             global_transformation=global_transformation,
                                             to_single_waveform=to_single_waveform,
                                             program_builder=subset_program_builder
                                             )
        
        
        elif (mode:=program_builder._stack[-1][0]) in {"iteration","repetition"}:
            # pt_dict = self.build_schedule(scope[0] if mode=="iteration" else {key:scope for key in self._channel_subsets.keys()})
            pt_dict = self.build_schedule(scope[0] if mode=="iteration" else scope,program_builder._scheduler_options)

            for key,subset_program_builder_tuple in program_builder._stack[-1][1].items():
                if mode=="iteration":
                    if isinstance(subset_program_builder_tuple[0],MultiProgramBuilder):
                        iterator = subset_program_builder_tuple[0]._with_iteration(*subset_program_builder_tuple[1:])
                    else:
                        iterator = subset_program_builder_tuple[0].with_iteration(*subset_program_builder_tuple[1:])
                    for itrep_builder in iterator:
                        pt_dict[key]._create_program(scope=itrep_builder.inner_scope(*scope),
                                                     measurement_mapping=measurement_mapping,
                                                     channel_mapping=channel_mapping,
                                                     global_transformation=global_transformation,
                                                     to_single_waveform=to_single_waveform,
                                                     program_builder=itrep_builder
                                                     )
        
                else:
                    if isinstance(subset_program_builder_tuple[0],MultiProgramBuilder):
                        iterator = subset_program_builder_tuple[0]._with_repetition(*subset_program_builder_tuple[1:])
                    else:
                        iterator = subset_program_builder_tuple[0].with_repetition(*subset_program_builder_tuple[1:])
                    for itrep_builder in iterator:
                        pt_dict[key]._create_program(scope=scope,
                                                     measurement_mapping=measurement_mapping,
                                                     channel_mapping=channel_mapping,
                                                     global_transformation=global_transformation,
                                                     to_single_waveform=to_single_waveform,
                                                     program_builder=itrep_builder
                                                     )
        # elif program_builder._stack[-1][0] in {"sequence",}:
        #     for key,subset_sequence_program_builder in program_builder._stack[-1][1].items():
        #         pt_dict[key]._create_program(scope=scope,
        #                                      measurement_mapping=measurement_mapping,
        #                                      channel_mapping=channel_mapping,
        #                                      global_transformation=global_transformation,
        #                                      to_single_waveform=to_single_waveform,
        #                                      program_builder=subset_sequence_program_builder
        #                                      )
        else:
            raise NotImplementedError()


def get_symbolic_vals_with_conditions_from_dict(start_time_by_scheduled: Dict[Scheduled, ExpressionScalar],
                                                      ch_subset: Set[ChannelID],
                                                      min_val:bool=True) -> Dict[ChannelID,ExpressionScalar]:
    # conditions = []  # To store conditions for Piecewise
    
    if len(start_time_by_scheduled)==0:
        return {ch: ExpressionScalar(0.) for ch in ch_subset}
    
    conditions_by_channel = {ch:[] for ch in ch_subset}
    
    # List of objects and their expr1
    scheduled_list = list(start_time_by_scheduled.keys())
    
    # Compare each expr1 with others and create conditions
    for i, scheduled in enumerate(scheduled_list):
        current_conditions = []
        
        # For each object, create conditions against all other objects
        for j, other_scheduled in enumerate(scheduled_list):
            if i != j:
                if min_val:
                    current_conditions.append(sp.Lt(start_time_by_scheduled[scheduled].underlying_expression,
                                                    start_time_by_scheduled[other_scheduled].underlying_expression))
                else:
                    current_conditions.append(sp.Gt(start_time_by_scheduled[scheduled].underlying_expression+scheduled.pt.duration.underlying_expression,
                                                    start_time_by_scheduled[other_scheduled].underlying_expression+other_scheduled.pt.duration.underlying_expression))
        
        # Combine all conditions (AND logic) for object i to be the smallest
        combined_condition = sp.And(*current_conditions)
        if min_val:
            for ch in ch_subset:
                if scheduled.pre_gap_volt is GAP_VOLT.DEFAULT:
                    
                    conditions_by_channel[ch].append((scheduled.pt.initial_values[ch].underlying_expression, combined_condition))
                elif scheduled.pre_gap_volt is GAP_VOLT.ZERO:
                    conditions_by_channel[ch].append((0, combined_condition))
                else:
                    raise NotImplementedError()
        else:
            for ch in ch_subset:
                if scheduled.post_gap_volt is GAP_VOLT.LAST:
                    conditions_by_channel[ch].append((scheduled.pt.final_values[ch].underlying_expression, combined_condition))
                elif scheduled.post_gap_volt is GAP_VOLT.ZERO:
                    conditions_by_channel[ch].append((0, combined_condition))
                else:
                    raise NotImplementedError()
    
    # If none of the conditions are met (i.e., for cases of equality), return the expr2 of the first object as default
    default_condition = {}
    if min_val:
        for ch in ch_subset:
            if scheduled_list[0].pre_gap_volt is GAP_VOLT.DEFAULT:
                default_condition[ch] = (scheduled_list[0].pt.initial_values[ch].underlying_expression, True)
            elif scheduled_list[0].pre_gap_volt is GAP_VOLT.ZERO:
                default_condition[ch] = (0, True)
            else:
                raise NotImplementedError()
    else:
        for ch in ch_subset:
            if scheduled_list[0].post_gap_volt is GAP_VOLT.LAST:
                default_condition[ch] = (scheduled_list[0].pt.final_values[ch].underlying_expression, True)
            elif scheduled_list[0].post_gap_volt is GAP_VOLT.ZERO:
                default_condition[ch] = (0, True)
            else:
                raise NotImplementedError()
    
    
    # Create and return the Piecewise expression
    return {ch: ExpressionScalar(sp.Piecewise(*conditions_by_channel[ch], default_condition[ch])) for ch in ch_subset}



