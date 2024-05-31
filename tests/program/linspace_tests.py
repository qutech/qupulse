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

        self.program = LinSpaceIter(
            length=200,
            body=(LinSpaceHold(
                bases=(-1.,),
                factors=((0.01,),),
                duration_base=TimeType(10**6),
                duration_factors=None
            ),)
        )

        key = DepKey.from_voltages((0.01,), DEFAULT_INCREMENT_RESOLUTION)

        self.commands = [
            Set(0, -1.0, key),
            Wait(TimeType(10 ** 6)),
            LoopLabel(0, 199),
            Increment(0, 0.01, key),
            Wait(TimeType(10 ** 6)),
            LoopJmp(0)
        ]

    def test_program(self):
        program_builder = LinSpaceBuilder(('a',))
        program = self.pulse_template.create_program(program_builder=program_builder)
        self.assertEqual([self.program], program)

    def test_commands(self):
        commands = to_increment_commands([self.program])
        self.assertEqual(self.commands, commands)


class PlainCSDTest(TestCase):
    def setUp(self):
        hold = ConstantPT(10**6, {'a': '-1. + idx_a * 0.01', 'b': '-.5 + idx_b * 0.02'})
        scan_a = hold.with_iteration('idx_a', 200)
        self.pulse_template = scan_a.with_iteration('idx_b', 100)

        self.program = LinSpaceIter(length=100, body=(LinSpaceIter(
            length=200,
            body=(LinSpaceHold(
                bases=(-1., -0.5),
                factors=((0.0, 0.01),
                         (0.02, 0.0)),
                duration_base=TimeType(10**6),
                duration_factors=None
            ),)
        ),))

        key_0 = DepKey.from_voltages((0, 0.01,), DEFAULT_INCREMENT_RESOLUTION)
        key_1 = DepKey.from_voltages((0.02,), DEFAULT_INCREMENT_RESOLUTION)

        self.commands = [
            Set(0, -1.0, key_0),
            Set(1, -0.5, key_1),
            Wait(TimeType(10 ** 6)),

            LoopLabel(0, 199),
            Increment(0, 0.01, key_0),
            Wait(TimeType(10 ** 6)),
            LoopJmp(0),

            LoopLabel(1, 99),

            Increment(0, -2.0, key_0),
            Increment(1, 0.02, key_1),
            Wait(TimeType(10 ** 6)),

            LoopLabel(2, 199),
            Increment(0, 0.01, key_0),
            Wait(TimeType(10 ** 6)),
            LoopJmp(2),

            LoopJmp(1),
        ]

    def test_program(self):
        program_builder = LinSpaceBuilder(('a', 'b'))
        program = self.pulse_template.create_program(program_builder=program_builder)
        self.assertEqual([self.program], program)

    def test_increment_commands(self):
        commands = to_increment_commands([self.program])
        self.assertEqual(self.commands, commands)


class TiltedCSDTest(TestCase):
    def setUp(self):
        hold = ConstantPT(10**6, {'a': '-1. + idx_a * 0.01 + idx_b * 1e-3', 'b': '-.5 + idx_b * 0.02 - 3e-3 * idx_a'})
        scan_a = hold.with_iteration('idx_a', 200)
        self.pulse_template = scan_a.with_iteration('idx_b', 100)
        self.repeated_pt = self.pulse_template.with_repetition(42)

        self.program = LinSpaceIter(length=100, body=(LinSpaceIter(
            length=200,
            body=(LinSpaceHold(
                bases=(-1., -0.5),
                factors=((1e-3, 0.01),
                         (0.02, -3e-3)),
                duration_base=TimeType(10**6),
                duration_factors=None
            ),)
        ),))
        self.repeated_program = LinSpaceRepeat(body=(self.program,), count=42)

        key_0 = DepKey.from_voltages((1e-3, 0.01,), DEFAULT_INCREMENT_RESOLUTION)
        key_1 = DepKey.from_voltages((0.02, -3e-3), DEFAULT_INCREMENT_RESOLUTION)

        self.commands = [
            Set(0, -1.0, key_0),
            Set(1, -0.5, key_1),
            Wait(TimeType(10 ** 6)),

            LoopLabel(0, 199),
            Increment(0, 0.01, key_0),
            Increment(1, -3e-3, key_1),
            Wait(TimeType(10 ** 6)),
            LoopJmp(0),

            LoopLabel(1, 99),

            Increment(0, 1e-3 + -200 * 1e-2, key_0),
            Increment(1, 0.02 + -200 * -3e-3, key_1),
            Wait(TimeType(10 ** 6)),

            LoopLabel(2, 199),
            Increment(0, 0.01, key_0),
            Increment(1, -3e-3, key_1),
            Wait(TimeType(10 ** 6)),
            LoopJmp(2),

            LoopJmp(1),
        ]
        inner_commands = copy.deepcopy(self.commands)
        for cmd in inner_commands:
            if hasattr(cmd, 'idx'):
                cmd.idx += 1
        self.repeated_commands = [LoopLabel(0, 42)] + inner_commands + [LoopJmp(0)]

    def test_program(self):
        program_builder = LinSpaceBuilder(('a', 'b'))
        program = self.pulse_template.create_program(program_builder=program_builder)
        self.assertEqual([self.program], program)

    def test_repeated_program(self):
        program_builder = LinSpaceBuilder(('a', 'b'))
        program = self.repeated_pt.create_program(program_builder=program_builder)
        self.assertEqual([self.repeated_program], program)

    def test_increment_commands(self):
        commands = to_increment_commands([self.program])
        self.assertEqual(self.commands, commands)

    def test_repeated_increment_commands(self):
        commands = to_increment_commands([self.repeated_program])
        self.assertEqual(self.repeated_commands, commands)


class SingletLoadProcessing(TestCase):
    def setUp(self):
        wait = ConstantPT(10 ** 6, {'a': '-1. + idx_a * 0.01', 'b': '-.5 + idx_b * 0.02'})
        load_random = ConstantPT(10 ** 5, {'a': -.4, 'b': -.3})
        meas = ConstantPT(10 ** 5, {'a': 0.05, 'b': 0.06})

        singlet_scan = (load_random @ wait @ meas).with_iteration('idx_a', 200).with_iteration('idx_b', 100)
        self.pulse_template = singlet_scan

        self.program = LinSpaceIter(length=100, body=(LinSpaceIter(
            length=200,
            body=(
                LinSpaceHold(bases=(-0.4, -0.3), factors=(None, None), duration_base=TimeType(10 ** 5),
                             duration_factors=None),
                LinSpaceHold(bases=(-1., -0.5),
                             factors=((0.0, 0.01),
                                      (0.02, 0.0)),
                             duration_base=TimeType(10 ** 6),
                             duration_factors=None),
                LinSpaceHold(bases=(0.05, 0.06), factors=(None, None), duration_base=TimeType(10 ** 5),
                             duration_factors=None),
            )
        ),))

        key_0 = DepKey.from_voltages((0, 0.01,), DEFAULT_INCREMENT_RESOLUTION)
        key_1 = DepKey.from_voltages((0.02,), DEFAULT_INCREMENT_RESOLUTION)

        self.commands = [
            Set(0, -0.4),
            Set(1, -0.3),
            Wait(TimeType(10 ** 5)),
            Set(0, -1.0, key_0),
            Set(1, -0.5, key_1),
            Wait(TimeType(10 ** 6)),
            Set(0, 0.05),
            Set(1, 0.06),
            Wait(TimeType(10 ** 5)),

            LoopLabel(0, 199),
            Set(0, -0.4),
            Set(1, -0.3),
            Wait(TimeType(10 ** 5)),
            Increment(0, 0.01, key_0),
            Increment(1, 0.00, key_1),
            Wait(TimeType(10 ** 6)),
            Set(0, 0.05),
            Set(1, 0.06),
            Wait(TimeType(10 ** 5)),
            LoopJmp(0),

            LoopLabel(1, 99),

            Set(0, -0.4),
            Set(1, -0.3),
            Wait(TimeType(10 ** 5)),
            Increment(0, -2.0, key_0),
            Increment(1, 0.02, key_1),
            Wait(TimeType(10 ** 6)),
            Set(0, 0.05),
            Set(1, 0.06),
            Wait(TimeType(10 ** 5)),

            LoopLabel(2, 199),

            Set(0, -0.4),
            Set(1, -0.3),
            Wait(TimeType(10 ** 5)),
            Increment(0, 0.01, key_0),
            Increment(1, 0.00, key_1),
            Wait(TimeType(10 ** 6)),
            Set(0, 0.05),
            Set(1, 0.06),
            Wait(TimeType(10 ** 5)),

            LoopJmp(2),

            LoopJmp(1),
        ]

    def test_singlet_scan_program(self):
        program_builder = LinSpaceBuilder(('a', 'b'))
        program = self.pulse_template.create_program(program_builder=program_builder)
        self.assertEqual([self.program], program)

    def test_singlet_scan_commands(self):
        commands = to_increment_commands([self.program])
        self.assertEqual(self.commands, commands)


class TransformedRampTest(TestCase):
    def setUp(self):
        hold = ConstantPT(10 ** 6, {'a': '-1. + idx * 0.01'})
        self.pulse_template = hold.with_iteration('idx', 200)
        self.transformation = ScalingTransformation({'a': 2.0})

        self.program = LinSpaceIter(
            length=200,
            body=(LinSpaceHold(
                bases=(-2.,),
                factors=((0.02,),),
                duration_base=TimeType(10 ** 6),
                duration_factors=None
            ),)
        )

    def test_global_trafo_program(self):
        program_builder = LinSpaceBuilder(('a',))
        program = self.pulse_template.create_program(program_builder=program_builder,
                                                     global_transformation=self.transformation)
        self.assertEqual([self.program], program)

    def test_local_trafo_program(self):
        program_builder = LinSpaceBuilder(('a',))
        with self.assertRaises(NotImplementedError):
            # not implemented yet. This test should work as soon as its implemented
            program = self.pulse_template.create_program(program_builder=program_builder,
                                                         global_transformation=self.transformation,
                                                         to_single_waveform={self.pulse_template})
            self.assertEqual([self.program], program)


class TimeSweepTest(TestCase):
    def setUp(self,base_time=1e2,rep_factor=2):
        wait = ConstantPT(f'64*{base_time}*1e1*(1+idx_t)', {'a': '-1. + idx_a * 0.01', 'b': '-.5 + idx_b * 0.02'})
    
        random_constant = ConstantPT(10 ** 5, {'a': -.4, 'b': -.3})
        meas = ConstantPT(64*base_time, {'a': 0.05, 'b': 0.06})
    
        singlet_scan = (random_constant @ wait @ meas).with_iteration('idx_a', rep_factor*10*2)\
                                                      .with_iteration('idx_b', rep_factor*10)\
                                                      .with_iteration('idx_t', 10)
        self.pulse_template = singlet_scan
    
        
    def test_singlet_scan_program(self):
        program_builder = LinSpaceBuilder(('a', 'b'))
        program = self.pulse_template.create_program(program_builder=program_builder)
        # so far just a test to see if the program creation works at all.
        # self.assertEqual([self.program], program)