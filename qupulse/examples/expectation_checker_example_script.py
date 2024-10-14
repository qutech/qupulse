# -*- coding: utf-8 -*-
import numpy as np

from qupulse.pulses import PointPT, ConstantPT, RepetitionPT, ForLoopPT, TablePT,\
    FunctionPT, AtomicMultiChannelPT, SequencePT, MappingPT, ParallelConstantChannelPT
from qupulse.program.linspace import LinSpaceBuilder

from qupulse.plotting import plot
from qupulse.utils import to_next_multiple

from expectation_checker import ExpectationChecker, HDAWGAlazar

#%% Get Devices

# Get the Alazar and the HDAWG in hardcoded configuration (2V p-p range)
# Connect any Marker from the HDAWG to Trig in of the Alazar
# Connect one dummy channel from HDAWG to the external clock of the Alazar,
# then set 0.5V-10MHz-oscillator on this channel (e.g. in HDAWG-Webinterface)

ha = HDAWGAlazar("DEVXXXX","USB",)

#%% Example pulse definitions

# PulseTemplates must be defined on channels named as 'ZI0_X', X in A to H
# (ensure correct mapping to Alazar)
# Markers will be overwritten to play a marker on each channel to trigger the Alazar
# identifiers of the final PT will be the names of the plotted object


class ShortSingleRampTest():
    def __init__(self, base_time=1e3):
        hold = ConstantPT(base_time, {'a': '-1. + idx * 0.01'})
        pt = hold.with_iteration('idx', 200)
        self.pulse_template = MappingPT(pt,
                                        channel_mapping={'a':'ZI0_A',},
                                        identifier=self.__class__.__name__
                                        )

class ShortSingleRampTestWithPlay():
    def __init__(self,base_time=1e3+8):
        # init = PointPT([(1.0,1e4)],channel_names=('ZI0_MARKER_FRONT',))
        init = FunctionPT('1.0+1e-9*t',base_time,channel='ZI0_A_MARKER_FRONT')#.pad_to(to_next_multiple(1.0,16,4),)

        hold = ConstantPT(base_time, {'a': '-1. + idx * 0.01'})#.pad_to(to_next_multiple(1.0,16,4))
        pt = ParallelConstantChannelPT(init,dict(ZI0_A=0.))@(ParallelConstantChannelPT(hold,dict(ZI0_A_MARKER_FRONT=0.)).with_iteration('idx', 200))
        self.pulse_template = MappingPT(pt,
                                        channel_mapping={'a':'ZI0_A',},
                                        identifier=self.__class__.__name__
                                        )

class SequencedRepetitionTest():
    def __init__(self,base_time=1e2,rep_factor=2):
        wait = AtomicMultiChannelPT(
            ConstantPT(f'64*{base_time}', {'a': '-1. + idx_a * 0.01 + y_gain', }),
            ConstantPT(f'64*{base_time}', {'b': '-0.5 + idx_b * 0.02'})
            )
        
        dependent_constant = AtomicMultiChannelPT(
            ConstantPT(64*base_time, {'a': '-1.0 + y_gain'}),
            ConstantPT(64*base_time, {'b': '-0.5 + idx_b*0.02',}),            
            )
        
        dependent_constant2 = AtomicMultiChannelPT(
            ConstantPT(64*base_time, {'a': '-0.5 + y_gain'}),
            ConstantPT(64*base_time, {'b': '-0.3 + idx_b*0.02',}),            
            )
        
    
        pt = (dependent_constant @ dependent_constant2.with_repetition(rep_factor) @ (wait.with_iteration('idx_a', rep_factor))).with_iteration('idx_b', rep_factor)\

        self.pulse_template = MappingPT(pt,parameter_mapping=dict(y_gain=0.3,),
                                        channel_mapping={'a':'ZI0_A','b':'ZI0_C'},
                                        identifier=self.__class__.__name__
                                        )


class SteppedRepetitionTest():
    def __init__(self,base_time=1e2,rep_factor=2):

        wait = ConstantPT(f'64*{base_time}*(1+idx_t)', {'a': '-0.5 + idx_a * 0.15', 'b': '-.5 + idx_a * 0.3'})
        normal_pt = ParallelConstantChannelPT(FunctionPT("sin(t/1000)","t_sin",channel='a'),{'b':-0.2})
        amp_pt = ParallelConstantChannelPT("amp*1/8"*FunctionPT("sin(t/2000)","t_sin",channel='a'),{'b':-0.5})
        # amp_pt2 = ParallelConstantChannelPT("amp2*1/8"*FunctionPT("sin(t/1000)","t_sin",channel='a'),{'b':-0.5})
        amp_inner = ParallelConstantChannelPT(FunctionPT(f"(1+amp)*1/(2*{rep_factor})*sin(4*pi*t/t_sin)","t_sin",channel='a'),{'b':-0.5})
        amp_inner2 = ParallelConstantChannelPT(FunctionPT(f"(1+amp2)*1/(2*{rep_factor})*sin((1*freq)*4*pi*t/t_sin)+off/(2*{rep_factor})","t_sin",channel='a'),{'b':-0.3})

        pt = ((((normal_pt@amp_inner2).with_iteration('off', rep_factor)@normal_pt@wait)\
              .with_repetition(rep_factor))@amp_inner.with_iteration('amp', rep_factor))\
              .with_iteration('amp2', rep_factor).with_iteration('freq', rep_factor).with_iteration('idx_a',rep_factor)
       
        self.pulse_template = MappingPT(pt,parameter_mapping=dict(t_sin=64*base_time,idx_t=1,
                                                                  #idx_a=1,#freq=1,#amp2=1
                                                                  ),
                                        channel_mapping={'a':'ZI0_A','b':'ZI0_C'},
                                        identifier=self.__class__.__name__)

class TimeSweepTest():
    def __init__(self,base_time=1e2,rep_factor=3):
        wait = ConstantPT(f'64*{base_time}*(1+idx_t)',
                          {'a': '-1. + idx_a * 0.01', 'b': '-.5 + idx_b * 0.02'})
    
        random_constant = ConstantPT(64*base_time, {'a': -.4, 'b': -.3})
        meas = ConstantPT(64*base_time, {'a': 0.05, 'b': 0.06})
    
        singlet_scan = (SequencePT(random_constant,wait,meas,identifier='s')).with_iteration('idx_a', rep_factor)\
                                                      .with_iteration('idx_b', rep_factor)\
                                                      .with_iteration('idx_t', rep_factor)
                                                      
        self.pulse_template = MappingPT(singlet_scan,channel_mapping={'a':'ZI0_A','b':'ZI0_C'},
                                        identifier=self.__class__.__name__)


#%% Instantiate Checker

# select exemplary pulse
pulse = ShortSingleRampTest(1e3+8)
# pulse = ShortSingleRampTestWithPlay()
# pulse = SequencedRepetitionTest(1e3,4)
# pulse = SteppedRepetitionTest(1e2,3)
# pulse = TimeSweepTest(1e2,3)

# Define a program builder to test program with:
program_builder = LinSpaceBuilder(
    #set to True to ensure triggering at Program start if program starts with constant pulse
    play_marker_when_constant=True,
    #in case stepped repetitions are needed, insert variables here: 
    to_stepping_repeat={'example',},
    )

# Data will be saved as xr.Dataset in save_path
# data_offsets corrects for offsets in Alazar (not in saved data, only in plotting)
checker = ExpectationChecker(ha, pulse.pulse_template,
                             program_builder=program_builder,
                             save_path=SAVE_HERE,
                             data_offsets={'t_offset':-100.,'v_offset':0.008,'v_scale':0.9975}
                             )

assert float(pulse.pulse_template.duration) < 1e7, "Ensure you know what you're doing when recording long data"

#%% Run the checker

checker.run()
