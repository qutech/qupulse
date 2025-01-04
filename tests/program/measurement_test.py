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
        instructions = self.pulse_template.create_program(program_builder=builder)
        self.assertEqual(self.commands, instructions.commands)

    def test_table(self):
        table = to_table(self.commands)
        tab_a = table['A']
        tab_b = table['B']
        np.testing.assert_array_equal(self.table_a, tab_a)
        np.testing.assert_array_equal(self.table_b, tab_b)


class ComplexPulse(TestCase):
    def setUp(self):
        hold = ConstantPT(10 ** 6, {'a': 1}, measurements=[('A', 10, 100), ('B', '1 + ii * 2 + jj', '3 + ii + jj')])
        dyn_hold = ConstantPT('10 ** 6 - 4 * ii', {'a': 1}, measurements=[('A', 10, 100), ('B', '1 + ii * 2 + jj', '3 + ii + jj')])

        self.base_pulse_template = SequencePT(
            hold.with_repetition('n_inner')
                .with_iteration('ii', 'i_inner')
                .with_repetition('n_middle')
                .with_iteration('jj', 'i_outer'),
            measurements=[('A', 1, 100)]
        ).with_repetition('n_outer')

        self.small_parameters = dict(
            n_outer=2,
            i_outer=11,
            n_middle=3,
            i_inner=7,
            n_inner=5
        )

        # ~ 10**8 points
        self.large_parameters = dict(
            n_outer=2,
            i_outer=100,
            n_middle=10,
            i_inner=100,
            n_inner=5
        )

        self.small_pulse_template = self.base_pulse_template.with_mapping(self.small_parameters)

        self.small_commands = [
            LoopLabel(5, None, self.small_parameters['n_outer']),
            Measure('A', 1, 100),
            LoopLabel(4, 'jj', self.small_parameters['i_outer']),
            LoopLabel(3, None, self.small_parameters['n_middle']),
            LoopLabel(2, 'ii', self.small_parameters['i_inner']),
            LoopLabel(1, None, self.small_parameters['n_inner']),
            Measure('A', 10, 100),
            Measure('B', SimpleExpression(base=1, offsets={'ii': 2, 'jj': 1}), SimpleExpression(base=3, offsets={'ii': 1, 'jj': 1})),
            Wait(TimeType(10 ** 6)),
            LoopJmp(1),
            LoopJmp(2),
            LoopJmp(3),
            LoopJmp(4),
            LoopJmp(5),
        ]

    def test_commands(self):
        builder = MeasurementBuilder()
        commands = self.small_pulse_template.create_program(program_builder=builder)

        self.assertEqual(
            self.small_commands,
            commands.commands
        )

    def test_table(self):
        raise NotImplementedError()

    def test_fast_instruction_table(self):
        instructions = self.base_pulse_template.create_program(program_builder=MeasurementBuilder(), parameters=self.large_parameters)
        fast_instructions = FastInstructions.from_commands(1, instructions.commands, {})

        params = self.large_parameters
        all_mul = np.prod(list(params.values()))

        meas_lens = dict(zip(fast_instructions.measurement_names, fast_instructions.measurement_lengths))
        self.assertEqual(params['n_outer'] + all_mul, meas_lens['A'])
        self.assertEqual(all_mul, meas_lens['B'])

        table = fast_instructions.create_tables()

        raise NotImplementedError("TODO")


