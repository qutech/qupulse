import unittest
from unittest import mock

import numpy as np

from qupulse.pulses.arithmetic_pulse_template import ArithmeticAtomicPulseTemplate, ArithmeticPulseTemplate
from qupulse._program.waveforms import ArithmeticWaveform

from tests.pulses.sequencing_dummies import DummyPulseTemplate


class ArithmeticAtomicPulseTemplateTest(unittest.TestCase):
    def test_init(self):
        raise NotImplementedError()
    
    def test_requires_stop(self):
        raise NotImplementedError()
    
    def test_build_waveform(self):
        raise NotImplementedError()
    
    def test_integral(self):
        raise NotImplementedError()

    def test_duration(self):
        raise NotImplementedError()

    def test_code_operator(self):
        a = DummyPulseTemplate(duration=4, defined_channels={'a', 'b'}, parameter_names={'x', 'y'})
        b = DummyPulseTemplate(duration=4, defined_channels={'a', 'c'}, parameter_names={'x', 'z'})

        c = a + b
        self.assertEqual('+', c.arithmetic_operator)
        self.assertIs(c.lhs, a)
        self.assertIs(c.rhs, b)

        c = a - b
        self.assertEqual('-', c.arithmetic_operator)
        self.assertIs(c.lhs, a)
        self.assertIs(c.rhs, b)

    def test_simple_properties(self):
        a = DummyPulseTemplate(duration=4, defined_channels={'a', 'b'}, parameter_names={'x', 'y'}, measurements=[('m_a', 0, 1)], measurement_names={'m_a'})
        b = DummyPulseTemplate(duration=4, defined_channels={'a', 'c'}, parameter_names={'x', 'z'}, measurements=[('m_a', 0, 1)], measurement_names={'m_b'})

        c = ArithmeticAtomicPulseTemplate(a, '-', b, measurements=[('m_base', 0, 1)])
        self.assertEqual({'a', 'b', 'c'}, c.defined_channels)
        self.assertEqual({'x', 'y', 'z'}, c.parameter_names)
        self.assertEqual({'m_base', 'm_a', 'm_b'}, c.measurement_names)
        self.assertIs(c.lhs, a)
        self.assertIs(c.rhs, b)


class ArithmeticPulseTemplateTest(unittest.TestCase):
    def test_init(self):
        raise NotImplementedError()

    def test_parse_operand(self):
        raise NotImplementedError()

    def test_get_scalar_value(self):
        raise NotImplementedError()

    def test_internal_create_program(self):
        raise NotImplementedError()

    def test_integral(self):
        raise NotImplementedError()

    def test_simple_attributes(self):
        raise NotImplementedError()
