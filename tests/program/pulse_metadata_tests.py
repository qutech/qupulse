import copy
import unittest
from unittest import TestCase

from qupulse.pulses import *
from qupulse.program.linspace import *
from qupulse.program.transformation import *


class PulseMetaDataTest(TestCase):
    def setUp(self):
        hold_a = ConstantPT(10 ** 6, {'a': '-1. + idx * 0.01'})
        hold_b = ConstantPT(10 ** 6, {'a': '-0.2 + idx * 0.005'})
        hold_combined = SequencePT(hold_a,hold_b,identifier='hold_pt')
        hold_2 = ConstantPT(10 ** 6, {'a': '-0.5'})
        play_arbitrary = FunctionPT("tanh(a*t**2 + b*t + c) * sin(b*t + c) + cos(a*t)/2",192*1e5,channel="a")
        
        self.pulse_template = (hold @ play_arbitrary @ hold_2).with_iteration('idx', 200)
        
        self.pulse_metadata = {
            'hold_pt': PTMetaData(True,2.0),
            hold_2: PTMetaData(to_single_waveform=False,minimal_sample_rate=1e-10,),
            play_arbitrary: PTMetaData(False,1e-3)
            }
        
    def test_program(self):
        program_builder = LinSpaceBuilder(('a',))
        program = self.pulse_template.create_program(
            program_builder=program_builder,
            metadata=self.pulse_metadata
            )
