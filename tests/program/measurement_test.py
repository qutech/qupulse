import copy
import unittest
from unittest import TestCase

import numpy as np

from qupulse.pulses import *
from qupulse.program.measurement import *


class SingleRampTest(TestCase):
    def setUp(self):
        hold = ConstantPT(10 ** 6, {'a': '-1. + idx * 0.01'}, measurements=[('A', 10, 100), ('B', '1 + idx * 2', 200)])
        self.pulse_template = hold.with_iteration('idx', 200)

        self.commands = [
            LoopLabel(1, 'idx', 200),
            Measure('A', 10, 100),
            Measure('B', SimpleExpression(base=1, offsets={'idx': 2}), 200),
            Wait(TimeType(10 ** 6)),
            LoopJmp(1)
        ]

        self.table_a = np.array([(10 + 10**6 * idx, 100) for idx in range(200)])
        self.table_b = np.array([(1 + idx * 2 + 10**6 * idx, 200) for idx in range(200)])

    def test_commands(self):
        builder = MeasurementBuilder()
        commands = self.pulse_template.create_program(program_builder=builder)
        self.assertEqual(self.commands, commands)

    def test_table(self):
        table = to_table(self.commands)
        tab_a = table['A']
        tab_b = table['B']
        np.testing.assert_array_equal(self.table_a, tab_a)
        np.testing.assert_array_equal(self.table_b, tab_b)


