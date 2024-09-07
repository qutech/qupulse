import copy
import unittest
from unittest import TestCase

from qupulse.pulses import *
from qupulse.program.linspace import *
from qupulse.program.transformation import *

class SingleRampTest(TestCase):
    def setUp(self):
        hold = ConstantPT(10 ** 6, {'a': '-1. + idx * 0.01'})
        self.pulse_template = hold.with_iteration('idx', 200)

        self.program = LinSpaceTopLevel(body=(LinSpaceIter(
            to_be_stepped=False,
            length=200,
            body=(LinSpaceHold(
                channels=('a',),
                bases={'a': -1.0},
                factors={'a': (0.01,)},
                duration_base=TimeType(10**6),
                duration_factors=None
            ),)
        ),))

        key = DepKey.from_voltages((0.01,), DEFAULT_INCREMENT_RESOLUTION)

        self.commands = [
            Set('a',ResolutionDependentValue((),(),-1.0),key=key),
            Wait(TimeType(10 ** 6)),
            LoopLabel(0, 199),
            Increment('a',ResolutionDependentValue((0.01,),(1,),0.0),key=key),
            Wait(TimeType(10 ** 6)),
            LoopJmp(0)
        ]

    def test_program(self):
        program_builder = LinSpaceBuilder()
        program = self.pulse_template.create_program(program_builder=program_builder)
        self.assertEqual([self.program], program)

    def test_commands(self):
        program_builder = LinSpaceBuilder()
        program = self.pulse_template.create_program(program_builder=program_builder)
        self.commands_to_test = to_increment_commands(program)
        self.assertEqual(self.commands, self.commands_to_test)


class TimeSweepTest(TestCase):
    def setUp(self,base_time=1e2,rep_factor=2):
        wait = ConstantPT(f'64*{base_time}*1e1*(1+idx_t)',
                          {'a': '-1. + idx_a * 0.01', 'b': '-.5 + idx_b * 0.02'})
    
        random_constant = ConstantPT(10 ** 5, {'a': -.4, 'b': -.3})
        meas = ConstantPT(64*base_time, {'a': 0.05, 'b': 0.06})
    
        singlet_scan = (random_constant @ wait @ meas).with_iteration('idx_a', rep_factor*10*2)\
                                                      .with_iteration('idx_b', rep_factor*10)\
                                                      .with_iteration('idx_t', 10)
        self.pulse_template = singlet_scan
    
        
    def test_program(self):
        program_builder = LinSpaceBuilder()
        self._program_to_test = self.pulse_template.create_program(program_builder=program_builder)
        
    def test_commands(self):
        program_builder = LinSpaceBuilder()
        self._program_to_test = self.pulse_template.create_program(program_builder=program_builder)
        commands = to_increment_commands(self._program_to_test)
        # so far just a test to see if the program creation works at all.
        # self.assertEqual([self.program], program)
        
        
class SequencedIterationTest(TestCase):
    def setUp(self,base_time=1e2,rep_factor=2):
        wait = AtomicMultiChannelPT(
            ConstantPT(f'64*{base_time}*1e1', {'a': '-1. + idx_a * 0.01 + y_gain', }),
            ConstantPT(f'64*{base_time}*1e1', {'b': '-.5 + idx_b * 0.02'})
            )
        
        dependent_constant = AtomicMultiChannelPT(
            ConstantPT(10 ** 5, {'a': -.3}),
            ConstantPT(10 ** 5, {'b': 'idx_b*0.02',}),            
            )
        
        dependent_constant2 = AtomicMultiChannelPT(
            ConstantPT(2*10 ** 5, {'a': '-.3+idx_b*0.01'}),
            ConstantPT(2*10 ** 5, {'b': 'idx_b*0.01',}),            
            )
    
        pt = (dependent_constant @ dependent_constant2 @ (wait.with_iteration('idx_a', rep_factor*10*2)) \
              @ dependent_constant2).with_iteration('idx_b', rep_factor*10)\

        self.pulse_template = MappingPT(pt,parameter_mapping=dict(y_gain=0.3,))
        
    def test_program(self):
        program_builder = LinSpaceBuilder()
        self._program_to_test = self.pulse_template.create_program(program_builder=program_builder)
        
    def test_commands(self):
        program_builder = LinSpaceBuilder()
        self._program_to_test = self.pulse_template.create_program(program_builder=program_builder)
        commands = to_increment_commands(self._program_to_test)
        # so far just a test to see if the program creation works at all.
        # self.assertEqual([self.program], program)
    

class SequencedIterationTest(TestCase):
    def setUp(self,base_time=1e2,rep_factor=2):
        wait = AtomicMultiChannelPT(
            ConstantPT(f'64*{base_time}*1e1', {'a': '-1. + idx_a * 0.01 + y_gain', }),
            ConstantPT(f'64*{base_time}*1e1', {'b': '-.5 + idx_b * 0.02'})
            )
        
        dependent_constant = AtomicMultiChannelPT(
            ConstantPT(10 ** 5, {'a': -.3}),
            ConstantPT(10 ** 5, {'b': 'idx_b*0.02',}),            
            )
        
        dependent_constant2 = AtomicMultiChannelPT(
            ConstantPT(2*10 ** 5, {'a': '-.3+idx_b*0.01'}),
            ConstantPT(2*10 ** 5, {'b': 'idx_b*0.01',}),
            measurements=[('c',0,10**5)]    
            )
    
        pt = (dependent_constant @ dependent_constant2 @ (wait.with_iteration('idx_a', rep_factor*10*2)) \
              @ dependent_constant2).with_iteration('idx_b', rep_factor*10)\
        
        step_len = 10**5*5 + rep_factor*10*2*64*base_time*1e1
            
        self.measurements = {
            'c': (
                np.repeat(np.arange(0.,(rep_factor*10)*step_len,step_len),2)+np.array([10**5,step_len-2*10**5]*rep_factor*10),
                np.ones(rep_factor*10*2)*10**5
                )
            }
            
        self.pulse_template = MappingPT(pt,parameter_mapping=dict(y_gain=0.3,))
        
    def test_measurements(self):
        program_builder = LinSpaceBuilder()
        self._program_to_test = self.pulse_template.create_program(program_builder=program_builder)
        self.assertIsNone(np.testing.assert_array_equal(self._program_to_test[0].get_measurement_windows()['c'], self.measurements['c']))


class AmplitudeSweepTest(TestCase):
    def setUp(self,rep_factor=2):

        normal_pt = FunctionPT("sin(t/100)","t_sin",channel='a')
        amp_pt = "amp*1/8"*FunctionPT("sin(t/1000)","t_sin",channel='a')
        
        pt = (normal_pt@amp_pt@normal_pt@amp_pt@amp_pt@normal_pt).with_iteration('amp', rep_factor)
        self.pulse_template = MappingPT(pt,parameter_mapping=dict(t_sin=64e2,))
        
    def test_program(self):
        program_builder = LinSpaceBuilder()
        self._program_to_test = self.pulse_template.create_program(program_builder=program_builder)
        
    def test_commands(self):
        program_builder = LinSpaceBuilder()
        self._program_to_test = self.pulse_template.create_program(program_builder=program_builder)
        commands = to_increment_commands(self._program_to_test)
        
        # so far just a test to see if the program creation works at all.
        # self.assertEqual([self.program], program)
        

class SteppedRepetitionTest(TestCase):
    def setUp(self,base_time=1e2,rep_factor=2):

        wait = ConstantPT(f'64*{base_time}*1e1*(1+idx_t)', {'a': '-0.5 + idx_a * 0.15', 'b': '-.5 + idx_a * 0.3'})
        normal_pt = ParallelConstantChannelPT(FunctionPT("sin(t/1000)","t_sin",channel='a'),{'b':-0.2})
        amp_pt = ParallelConstantChannelPT("amp*1/8"*FunctionPT("sin(t/1000)","t_sin",channel='a'),{'b':-0.5})
        # amp_pt2 = ParallelConstantChannelPT("amp2*1/8"*FunctionPT("sin(t/1000)","t_sin",channel='a'),{'b':-0.5})
        amp_inner = ParallelConstantChannelPT(FunctionPT(f"(1+amp)*1/(2*{rep_factor})*sin(4*pi*t/t_sin)","t_sin",channel='a'),{'b':-0.5})
        amp_inner2 = ParallelConstantChannelPT(FunctionPT(f"(1+amp2)*1/(2*{rep_factor})*sin((1*freq)*4*pi*t/t_sin)+off/(2*{rep_factor})","t_sin",channel='a'),{'b':-0.3})

        pt = (((normal_pt@amp_inner2).with_iteration('off', rep_factor)@normal_pt@wait)\
              .with_repetition(rep_factor)@amp_inner.with_iteration('amp', rep_factor))\
              .with_iteration('amp2', rep_factor).with_iteration('freq', rep_factor).with_iteration('idx_a',rep_factor)
       
        self.pulse_template = MappingPT(pt,parameter_mapping=dict(t_sin=64e2,idx_t=1,))
        
    def test_program(self):
        program_builder = LinSpaceBuilder(to_stepping_repeat={'amp','amp2','off','freq'},)
        self._program_to_test = self.pulse_template.create_program(program_builder=program_builder)
        
    def test_commands(self):
        program_builder = LinSpaceBuilder(to_stepping_repeat={'amp','amp2','off','freq'},)
        self._program_to_test = self.pulse_template.create_program(program_builder=program_builder)
        commands = to_increment_commands(self._program_to_test)
        # so far just a test to see if the program creation works at all.
        # self.assertEqual([self.program], program)
        

class CombinedSweepTest(TestCase):
    def setUp(self,base_time=1e2,rep_factor=2):

        wait = ConstantPT(f'64*{base_time}*1e1*(1+idx_t)', {'a': f'-1. + idx_a * 0.5/{rep_factor}', 'b': f'-.5 + idx_a * 0.8/{rep_factor}'})
        normal_pt = ParallelConstantChannelPT(FunctionPT("sin(t/2000)","t_sin",channel='a'),{'b':-0.2})
        amp_pt = ParallelConstantChannelPT(f"amp*1/1.5 * 1/{rep_factor}"*FunctionPT("sin(t/2000)","t_sin",channel='a'),{'b':-0.5})

        pt = (normal_pt@amp_pt@normal_pt@wait@amp_pt@amp_pt@normal_pt)\
            .with_iteration('amp', rep_factor)\
            .with_iteration('idx_a', rep_factor)\
            .with_iteration('idx_t', rep_factor)
            
        self.pulse_template = MappingPT(pt,parameter_mapping=dict(t_sin=64e2,))
        
    def test_program(self):
        program_builder = LinSpaceBuilder()
        self._program_to_test = self.pulse_template.create_program(program_builder=program_builder)
        
    def test_commands(self):
        program_builder = LinSpaceBuilder()
        self._program_to_test = self.pulse_template.create_program(program_builder=program_builder)
        commands = to_increment_commands(self._program_to_test)
        # so far just a test to see if the program creation works at all.
        # self.assertEqual([self.program], program)