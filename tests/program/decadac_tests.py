from unittest import TestCase

from qupulse.pulses import *
from qupulse.program.decadac import *


class DecaDacProgramBuilderTests(TestCase):
    def test_single_channel_ramp(self):
        hold = ConstantPT(10**6, {'a': '-1. + idx * 0.01'})
        ramp = hold.with_iteration('idx', 200)

        program_builder = DecaDACASCIIBuilder.from_channel_dict({'a': 0})
        program = ramp.create_program(program_builder=program_builder)

        raise NotImplementedError('the rest of the owl')
