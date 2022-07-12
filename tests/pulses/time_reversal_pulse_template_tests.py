import unittest

from qupulse.pulses.time_reversal_pulse_template import TimeReversalPulseTemplate
from qupulse.utils.types import TimeType
from qupulse.expressions import ExpressionScalar

from tests.pulses.sequencing_dummies import DummyPulseTemplate
from tests.serialization_tests import SerializableTests


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

