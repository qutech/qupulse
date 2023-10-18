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
            body=(LinSpaceSet(
                bases=(-1.,),
                factors=((0.01,),),
                duration_base=TimeType(10**6),
                duration_factors=None
            ),)
        )

        self.assertEqual([expected], program)
