import unittest

from qupulse.pulses import FunctionPT, TimeExtensionPT, SingleWFTimeExtensionPT
from qupulse.program.loop import LoopBuilder
from qupulse.program.linspace import LinSpaceBuilder

class TimeExtensionPulseTemplateTests(unittest.TestCase):
    
    def setUp(self):
        self.main_pt = FunctionPT("tanh(a*t**2 + b*t + c) * sin(b*t + c) + cos(a*t)/2",192*1e1,channel="a")
        
        self.extend_prior = SingleWFTimeExtensionPT(self.main_pt,"t_prior","t_z")
        self.extend_posterior = SingleWFTimeExtensionPT(self.main_pt,"t_z","t_posterior")
        self.extend_both = SingleWFTimeExtensionPT(self.main_pt,"t_prior","t_posterior")
        
        self.parameters = dict(t_prior=256.,t_posterior=512.,t_z=0.,
                          a=1.,b=0.5,c=1.
                          )
        
        self.sequenced_pt = self.extend_prior @ self.extend_posterior @ self.extend_both
        
        self.sequenced_extended_pt = TimeExtensionPT(self.sequenced_pt,"t_prior","t_posterior")
        
    def test_loopbuilder(self):
        self.extend_both.create_program(program_builder=LoopBuilder(),
                                        parameters=self.parameters)
        
        self.sequenced_pt.create_program(program_builder=LoopBuilder(),
                                          parameters=self.parameters)
        
        self.sequenced_extended_pt.create_program(program_builder=LoopBuilder(),
                                                  parameters=self.parameters)

    def test_linspacebuilder(self):
        self.extend_both.create_program(program_builder=LinSpaceBuilder(('a',)),
                                        parameters=self.parameters)
        
        self.sequenced_pt.create_program(program_builder=LinSpaceBuilder(('a',)),
                                         parameters=self.parameters)
        
        self.sequenced_extended_pt.create_program(program_builder=LinSpaceBuilder(('a',)),
                                                  parameters=self.parameters)

    def test_quantities(self):
        
        self.assertEqual(self.extend_both.defined_channels, self.main_pt.defined_channels)
        
        
        duration_extended = self.extend_prior.duration.evaluate_in_scope(self.parameters)
        duration_summed = self.main_pt.duration.evaluate_in_scope(self.parameters)\
            + self.parameters['t_prior']
        self.assertEqual(duration_extended, duration_summed)
        
        duration_extended = self.extend_posterior.duration.evaluate_in_scope(self.parameters)
        duration_summed = self.main_pt.duration.evaluate_in_scope(self.parameters)\
            + self.parameters['t_posterior']
        self.assertEqual(duration_extended, duration_summed)
        
        duration_extended = self.extend_both.duration.evaluate_in_scope(self.parameters)
        duration_summed = self.main_pt.duration.evaluate_in_scope(self.parameters)\
            + self.parameters['t_prior'] + self.parameters['t_posterior']
        self.assertEqual(duration_extended, duration_summed)