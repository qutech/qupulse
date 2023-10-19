from unittest import TestCase

from qupulse.pulses import *
from qupulse.program.linspace import *


class IdxProgramBuilderTests(TestCase):
    def test_single_channel_ramp(self):
        hold = ConstantPT(10**6, {'a': '-1. + idx * 0.01'})
        ramp = hold.with_iteration('idx', 200)

        program_builder = LinSpaceBuilder.from_channel_dict({'a': 0})
        program = ramp.create_program(program_builder=program_builder)

        expected = LinSpaceIter(
            length=200,
            body=(LinSpaceHold(
                bases=(-1.,),
                factors=((0.01,),),
                duration_base=TimeType(10**6),
                duration_factors=None
            ),)
        )

        self.assertEqual([expected], program)

    def test_single_ramp_increment_commands(self):
        program = LinSpaceIter(
            length=200,
            body=(LinSpaceHold(
                bases=(-1.,),
                factors=((0.01,),),
                duration_base=TimeType(10 ** 6),
                duration_factors=None
            ),)
        )

        commands = to_increment_commands([program])

        expected = [
            Set(0, -1.0),
            Wait(TimeType(10 ** 6)),
            LoopLabel(0, 199),
            Increment(0, 0.01, DepKey.from_voltages((0.01,), DEFAULT_RESOLUTION)),
            Wait(TimeType(10 ** 6)),
            LoopJmp(0)
        ]
        self.assertEqual(expected, commands)

    def test_csd_program(self):
        hold = ConstantPT(10**6, {'a': '-1. + idx_a * 0.01', 'b': '-.5 + idx_b * 0.02'})
        scan_a = hold.with_iteration('idx_a', 200)
        csd = scan_a.with_iteration('idx_b', 100)

        program_builder = LinSpaceBuilder.from_channel_dict({'a': 0, 'b': 1})
        program = csd.create_program(program_builder=program_builder)

        expected = LinSpaceIter(length=100, body=(LinSpaceIter(
            length=200,
            body=(LinSpaceHold(
                bases=(-1., -0.5),
                factors=((0.0, 0.01),
                         (0.02, 0.0)),
                duration_base=TimeType(10**6),
                duration_factors=None
            ),)
        ),))

        self.assertEqual([expected], program)

    def test_csd_increment_commands(self):
        program = LinSpaceIter(length=100, body=(LinSpaceIter(
            length=200,
            body=(LinSpaceHold(
                bases=(-1., -0.5),
                factors=((0.0, 0.01),
                         (0.02, 0.0)),
                duration_base=TimeType(10 ** 6),
                duration_factors=None
            ),)
        ),))

        commands = to_increment_commands([program])

        expected = [
            Set(0, -1.0, DepKey.from_voltages((0, 0.01,), DEFAULT_RESOLUTION)),
            Set(1, -0.5, DepKey.from_voltages((0.02,), DEFAULT_RESOLUTION)),
            Wait(TimeType(10 ** 6)),

            LoopLabel(0, 199),
            Increment(0, 0.01, DepKey.from_voltages((0, 0.01,), DEFAULT_RESOLUTION)),
            Wait(TimeType(10 ** 6)),
            LoopJmp(0),

            LoopLabel(1, 99),

            Increment(0, -2.0, DepKey.from_voltages((0, 0.01,), DEFAULT_RESOLUTION)),
            Increment(1, 0.02, DepKey.from_voltages((0.02,), DEFAULT_RESOLUTION)),
            Wait(TimeType(10 ** 6)),

            LoopLabel(2, 199),
            Increment(0, 0.01, DepKey.from_voltages((0, 0.01,), DEFAULT_RESOLUTION)),
            Wait(TimeType(10 ** 6)),
            LoopJmp(2),

            LoopJmp(1),
        ]
        self.assertEqual(expected, commands)


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

        key_0 = DepKey.from_voltages((0, 0.01,), DEFAULT_RESOLUTION)
        key_1 = DepKey.from_voltages((0.02,), DEFAULT_RESOLUTION)

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
        program_builder = LinSpaceBuilder.from_channel_dict({'a': 0, 'b': 1})
        program = self.pulse_template.create_program(program_builder=program_builder)
        self.assertEqual([self.program], program)

    def test_singlet_scan_commands(self):
        commands = to_increment_commands([self.program])
        self.assertEqual(self.commands, commands)

