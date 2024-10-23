# -*- coding: utf-8 -*-
import numpy as np

from qupulse.pulses import PointPT, ConstantPT, RepetitionPT, ForLoopPT, TablePT,\
    FunctionPT, AtomicMultiChannelPT, SequencePT, MappingPT, ParallelConstantChannelPT,\
    SchedulerPT
from qupulse.program.linspace import LinSpaceBuilder, to_increment_commands
from qupulse.program.multi import MultiProgramBuilder
from qupulse.program.loop import LoopBuilder

from qupulse.plotting import plot
from qupulse.utils import to_next_multiple

from expectation_checker_meta import ExpectationChecker, HDAWGAlazar

#%% Get Devices

# Get the Alazar and the HDAWG in hardcoded configuration (2V p-p range)
# Connect any Marker from the HDAWG to Trig in of the Alazar
# Connect one dummy channel from HDAWG to the external clock of the Alazar,
# then set 0.5V-10MHz-oscillator on this channel (e.g. in HDAWG-Webinterface)

ha = HDAWGAlazar("DEVXXXX","USB",)

#%% Example pulse definitions

# PulseTemplates must be defined on channels named as 'ZI0_X', X in A to H
# and possibly marker channel(s), as denoted +'_MARKER_FRONT'
# channel subsets {AB,CD,EF,GH} to be referenced as f'expcheck_{i}', i in range(4)
# (ensure correct mapping to Alazar)
# Markers need to be added on the according channel to trigger the alazar

        
class MultiSchedule():
    def __init__(self,base_time=1e2,rep_factor=3):
        
        init_chs = {'ZI0_A','ZI0_B','ZI0_A_MARKER_FRONT','ZI0_B_MARKER_FRONT',}
        manip_chs = {'ZI0_C','ZI0_D','ZI0_C_MARKER_FRONT','ZI0_D_MARKER_FRONT',}
        
        zone1 = {'expcheck_0': init_chs, 'expcheck_1': manip_chs,}
        
        init_chs_2 = {'ZI0_E','ZI0_F','ZI0_E_MARKER_FRONT','ZI0_F_MARKER_FRONT',}
        manip_chs_2 = {'ZI0_G','ZI0_H','ZI0_G_MARKER_FRONT','ZI0_H_MARKER_FRONT',}
        
        top_channel_structure = {
                                'zone1': zone1,
                                'expcheck_2': init_chs_2,
                                }
        
        markers_ab = {f'ZI0_{x}_MARKER_FRONT':1.0 for x in 'AB'}#|{f'ZI0_{x}_MARKER_BACK':0.0 for x in 'AB'}
        markers_cd = {f'ZI0_{x}_MARKER_FRONT':1.0 for x in 'CD'}#|{f'ZI0_{x}_MARKER_BACK':0.0 for x in 'CD'}
        markers_ef = {f'ZI0_{x}_MARKER_FRONT':1.0 for x in 'EF'}#|{f'ZI0_{x}_MARKER_BACK':0.0 for x in 'EF'}
        markers_ab_to_ef = {ch1:ch2 for ch1,ch2 in zip(markers_ab.keys(),markers_ef.keys())}
        
        some_pt = ParallelConstantChannelPT("amp*1/8"*FunctionPT("sin(t/100)","t_sin",channel='a'),{'b':-0.5}|markers_ab)
        some_pt2 = ParallelConstantChannelPT("amp*1/8"*FunctionPT("sin(t/100)","t_sin",channel='a'),{'b':-0.5}|markers_ab)
        some_other_pt2 = ParallelConstantChannelPT("amp*1/8"*FunctionPT("sin(t/200)","t_sin",channel='c'),{'d':-0.6}|markers_cd)

        
        schedule = SchedulerPT(zone1)
        top_schedule = SchedulerPT(top_channel_structure)
        
        s1 = schedule.add_pt(MappingPT(some_pt,channel_mapping=dict(a='ZI0_A',b='ZI0_B')), None)
        s2 = schedule.add_pt(MappingPT(some_pt2,channel_mapping=dict(a='ZI0_A',b='ZI0_B')),s1)
        s3 = schedule.add_pt(MappingPT(some_other_pt2,channel_mapping=dict(c='ZI0_C',d='ZI0_D')),s1,rel_time=80+16.)
        s3 = schedule.add_pt(MappingPT(some_other_pt2,channel_mapping=dict(c='ZI0_C',d='ZI0_D'),parameter_mapping=dict(t_sin="3.5*t_sin",amp='0.7')),s3,rel_time=4*80+16.)


        looped = ForLoopPT(schedule, 'amp', 5)
        # looped = RepetitionPT(schedule,5)
        
        n1 = top_schedule.add_pt(looped, None)
        n2 = top_schedule.add_pt(MappingPT(some_pt2,parameter_mapping={'amp':0.5,'t_sin':1e4},channel_mapping=dict(a='ZI0_E',b='ZI0_F')|markers_ab_to_ef), n1, ref_point="start",rel_time=-128.)
        # n2 = top_schedule.add_pt(MappingPT(some_pt2,parameter_mapping={'amp':0.5,'t_sin':1e4},channel_mapping=dict(a='ZI0_G',b='ZI0_H')), n1, ref_point="start",rel_time=192.)
                                             
        self.pulse_template = MappingPT(top_schedule,parameter_mapping=dict(t_sin=192.))
        self.channel_subsets = top_channel_structure
        

#%% Instantiate Checker

# select exemplary pulse
pulse = MultiSchedule()

# # Define a program builder to test program with:
default_program_builder = LinSpaceBuilder(
    #set to True to ensure triggering at Program start if program starts with constant pulse
    play_marker_when_constant=True,
    #in case stepped repetitions are needed, insert variables here: 
    # to_stepping_repeat={'example',},
    # to_stepping_repeat={'amp','amp2','off','freq'},
    )

program_builder = MultiProgramBuilder.from_mapping(default_program_builder, channel_subsets=pulse.channel_subsets)

# Data will be saved as xr.Dataset in save_path
# data_offsets corrects for offsets in Alazar (not in saved data, only in plotting)
checker = ExpectationChecker(ha, pulse.pulse_template,
                             program_builder=program_builder,
                             save_path=r"D:\2302_experiment_env\qupulse_dev_scripts\expcheck",
                             data_offsets={'t_offset':-100.,'v_offset':0.008,'v_scale':0.9975}
                             )

assert float(pulse.pulse_template.duration) < 1e7, "Ensure you know what you're doing when recording long data"

#%% Run the checker

checker.run()
