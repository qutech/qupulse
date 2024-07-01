import unittest

import numpy as np

from qupulse.pulses import ConstantPT, FunctionPT
from qupulse.plotting import render
from qupulse.pulses.time_reversal_pulse_template import TimeReversalPulseTemplate
from qupulse.utils.types import TimeType
from qupulse.expressions import ExpressionScalar
from qupulse.program.loop import LoopBuilder
from qupulse.program.linspace import LinSpaceBuilder, LinSpaceVM, to_increment_commands
from tests.pulses.sequencing_dummies import DummyPulseTemplate
from tests.serialization_tests import SerializableTests
from tests.program.linspace_tests import assert_vm_output_almost_equal

class TimeReversalPulseTemplateTests(unittest.TestCase):
    def test_simple_properties(self):
        inner = DummyPulseTemplate(identifier='d',
                                   defined_channels={'A', 'B'},
                                   duration=ExpressionScalar(42),
                                   integrals={'A': ExpressionScalar(4), 'B': ExpressionScalar('alpha')},
                                   parameter_names={'alpha', 'beta'})

        reversed_pt = TimeReversalPulseTemplate(inner, identifier='reverse')

        self.assertEqual(reversed_pt.duration, inner.duration)
        self.assertEqual(reversed_pt.parameter_names, inner.parameter_names)
        self.assertEqual(reversed_pt.integral, inner.integral)
        self.assertEqual(reversed_pt.defined_channels, inner.defined_channels)

        self.assertEqual(reversed_pt.identifier, 'reverse')

    def test_time_reversal_loop(self):
        inner = ConstantPT(4, {'a': 3}) @ FunctionPT('sin(t)', 5, channel='a')
        manual_reverse = FunctionPT('sin(5 - t)', 5, channel='a') @ ConstantPT(4, {'a': 3})
        time_reversed = TimeReversalPulseTemplate(inner)

        program = time_reversed.create_program(program_builder=LoopBuilder())
        manual_program = manual_reverse.create_program(program_builder=LoopBuilder())

        t, data, _ = render(program, 9 / 10)
        _, manual_data, _ = render(manual_program, 9 / 10)

        np.testing.assert_allclose(data['a'], manual_data['a'])

    def test_time_reversal_linspace(self):
        constant_pt = ConstantPT(4, {'a': '3.0 + x * 1.0 + y * -0.3'})
        function_pt = FunctionPT('sin(t)', 5, channel='a')
        reversed_function_pt = FunctionPT('sin(5 - t)', 5, channel='a')

        inner = (constant_pt @ function_pt).with_iteration('x', 6)
        inner_manual = (reversed_function_pt @ constant_pt).with_iteration('x', (5, -1, -1))

        outer = inner.with_time_reversal().with_iteration('y', 8)
        outer_man = inner_manual.with_iteration('y', 8)

        self.assertEqual(outer.duration, outer_man.duration)

        program = outer.create_program(program_builder=LinSpaceBuilder(channels=('a',)))
        manual_program = outer_man.create_program(program_builder=LinSpaceBuilder(channels=('a',)))

        commands = to_increment_commands(program)
        manual_commands = to_increment_commands(manual_program)

        manual_vm = LinSpaceVM(1)
        manual_vm.set_commands(manual_commands)
        manual_vm.run()

        vm = LinSpaceVM(1)
        vm.set_commands(commands)
        vm.run()

        assert_vm_output_almost_equal(self, manual_vm.history, vm.history)


class TimeReversalPulseTemplateSerializationTests(unittest.TestCase, SerializableTests):
    @property
    def class_to_test(self):
        return TimeReversalPulseTemplate

    def make_kwargs(self) -> dict:
        return dict(
            inner=DummyPulseTemplate(identifier='d',
                                     defined_channels={'A', 'B'},
                                     duration=ExpressionScalar(42),
                                     integrals={'A': ExpressionScalar(4), 'B': ExpressionScalar('alpha')},
                                     parameter_names={'alpha', 'beta'}),
        )

    def assert_equal_instance_except_id(self, lhs, rhs):
        return lhs._inner == rhs._inner

