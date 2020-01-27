import unittest
from unittest import mock
import warnings

import numpy as np
import sympy

from qupulse.expressions import ExpressionScalar
from qupulse.pulses.arithmetic_pulse_template import ArithmeticAtomicPulseTemplate, ArithmeticPulseTemplate,\
    ImplicitAtomicityInArithmeticPT, UnequalDurationWarningInArithmeticPT
from qupulse._program.waveforms import ArithmeticWaveform

from tests.pulses.sequencing_dummies import DummyPulseTemplate, DummyWaveform
from tests.pulses.pulse_template_tests import PulseTemplateStub
from tests.serialization_tests import SerializableTests


class ArithmeticAtomicPulseTemplateTest(unittest.TestCase):
    def test_init(self):
        lhs = DummyPulseTemplate(duration=4, defined_channels={'a', 'b'}, parameter_names={'x', 'y'})
        rhs = DummyPulseTemplate(duration=4, defined_channels={'a', 'c'}, parameter_names={'x', 'z'})

        non_atomic = PulseTemplateStub(duration=4)
        wrong_duration = DummyPulseTemplate(duration=3, defined_channels={'a', 'c'}, parameter_names={'x', 'z'})

        measurements = [('m1', 0, 1), ('m2', 1, 2)]

        with self.assertRaises(ValueError):
            ArithmeticAtomicPulseTemplate(lhs=lhs, rhs=rhs, arithmetic_operator='*')

        with self.assertWarns(ImplicitAtomicityInArithmeticPT):
            ArithmeticAtomicPulseTemplate(lhs=non_atomic, rhs=rhs, arithmetic_operator='+')

        with warnings.catch_warnings() as w:
            ArithmeticAtomicPulseTemplate(lhs=non_atomic, rhs=rhs, arithmetic_operator='+', silent_atomic=True)
            self.assertFalse(w)

        with self.assertWarns(UnequalDurationWarningInArithmeticPT):
            ArithmeticAtomicPulseTemplate(lhs=wrong_duration, rhs=rhs, arithmetic_operator='+')

        arith = ArithmeticAtomicPulseTemplate(lhs, '-', rhs, identifier='my_arith', measurements=measurements)
        self.assertIs(rhs, arith.rhs)
        self.assertIs(lhs, arith.lhs)
        self.assertEqual(measurements, arith.measurement_declarations)
        self.assertEqual('-', arith.arithmetic_operator)
        self.assertEqual('my_arith', arith.identifier)
    
    def test_requires_stop(self):
        lhs = DummyPulseTemplate(duration=4, defined_channels={'a', 'b'}, parameter_names={'x', 'y'})
        rhs = DummyPulseTemplate(duration=4, defined_channels={'a', 'c'}, parameter_names={'x', 'z'})

        arith = lhs + rhs
        with self.assertRaises(NotImplementedError):
            arith.requires_stop({}, {})
    
    def test_build_waveform(self):
        a = DummyPulseTemplate(duration=4, defined_channels={'a', 'b'}, parameter_names={'x', 'y'})
        b = DummyPulseTemplate(duration=4, defined_channels={'a', 'c'}, parameter_names={'x', 'z'})

        wf_a = DummyWaveform(duration=4)
        wf_b = DummyWaveform(duration=4)
        wf_arith = DummyWaveform(duration=4)
        wf_rhs_only = DummyWaveform(duration=4)

        arith = ArithmeticAtomicPulseTemplate(a, '-', b)

        parameters = dict(foo=8.)
        channel_mapping = dict(x='y', u='v')

        # channel a in both
        with mock.patch.object(a, 'build_waveform', return_value=wf_a) as build_a, mock.patch.object(b, 'build_waveform', return_value=wf_b) as build_b:
            with mock.patch('qupulse.pulses.arithmetic_pulse_template.ArithmeticWaveform', return_value=wf_arith) as wf_init:
                wf_init.rhs_only_map.__getitem__.return_value.return_value = wf_rhs_only
                self.assertIs(wf_arith, arith.build_waveform(parameters=parameters, channel_mapping=channel_mapping))
                wf_init.assert_called_once_with(wf_a, '-', wf_b)
                wf_init.rhs_only_map.__getitem__.assert_not_called()

            build_a.assert_called_once_with(parameters=parameters, channel_mapping=channel_mapping)
            build_b.assert_called_once_with(parameters=parameters, channel_mapping=channel_mapping)

        # only lhs
        with mock.patch.object(a, 'build_waveform', return_value=wf_a) as build_a, mock.patch.object(b, 'build_waveform', return_value=None) as build_b:
            with mock.patch('qupulse.pulses.arithmetic_pulse_template.ArithmeticWaveform', return_value=wf_arith) as wf_init:
                wf_init.rhs_only_map.__getitem__.return_value.return_value = wf_rhs_only
                self.assertIs(wf_a, arith.build_waveform(parameters=parameters, channel_mapping=channel_mapping))
                wf_init.assert_not_called()
                wf_init.rhs_only_map.__getitem__.assert_not_called()

            build_a.assert_called_once_with(parameters=parameters, channel_mapping=channel_mapping)
            build_b.assert_called_once_with(parameters=parameters, channel_mapping=channel_mapping)

        # only rhs
        with mock.patch.object(a, 'build_waveform', return_value=None) as build_a, mock.patch.object(b,
                                                                                                     'build_waveform',
                                                                                                     return_value=wf_b) as build_b:
            with mock.patch('qupulse.pulses.arithmetic_pulse_template.ArithmeticWaveform',
                            return_value=wf_arith) as wf_init:
                wf_init.rhs_only_map.__getitem__.return_value.return_value = wf_rhs_only
                self.assertIs(wf_rhs_only, arith.build_waveform(parameters=parameters, channel_mapping=channel_mapping))
                wf_init.assert_not_called()
                wf_init.rhs_only_map.__getitem__.assert_called_once_with('-')
                wf_init.rhs_only_map.__getitem__.return_value.assert_called_once_with(wf_b)

            build_a.assert_called_once_with(parameters=parameters, channel_mapping=channel_mapping)
            build_b.assert_called_once_with(parameters=parameters, channel_mapping=channel_mapping)

    def test_integral(self):
        integrals_lhs = dict(a=ExpressionScalar('a_lhs'), b=ExpressionScalar('b'))
        integrals_rhs = dict(a=ExpressionScalar('a_rhs'), c=ExpressionScalar('c'))

        lhs = DummyPulseTemplate(duration=4, defined_channels={'a', 'b'},
                                 parameter_names={'x', 'y'}, integrals=integrals_lhs)
        rhs = DummyPulseTemplate(duration=4, defined_channels={'a', 'c'},
                                 parameter_names={'x', 'z'}, integrals=integrals_rhs)

        expected_plus = dict(a=ExpressionScalar('a_lhs + a_rhs'),
                             b=ExpressionScalar('b'),
                             c=ExpressionScalar('c'))
        expected_minus = dict(a=ExpressionScalar('a_lhs - a_rhs'),
                              b=ExpressionScalar('b'),
                              c=ExpressionScalar('-c'))
        self.assertEqual(expected_plus, (lhs + rhs).integral)
        self.assertEqual(expected_minus, (lhs - rhs).integral)

    def test_duration(self):
        lhs = DummyPulseTemplate(duration=ExpressionScalar('x'), defined_channels={'a', 'b'}, parameter_names={'x', 'y'})
        rhs = DummyPulseTemplate(duration=ExpressionScalar('y'), defined_channels={'a', 'c'}, parameter_names={'x', 'z'})
        arith = lhs - rhs
        self.assertEqual(sympy.Max('x', 'y'), arith.duration.underlying_expression)

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


class ArithmeticAtomicPulseTemplateSerializationTest(SerializableTests, unittest.TestCase):
    @property
    def class_to_test(self):
        return ArithmeticAtomicPulseTemplate

    def make_kwargs(self):
        return {
            'rhs': DummyPulseTemplate(duration=ExpressionScalar('x')),
            'lhs': DummyPulseTemplate(duration=ExpressionScalar('x')),
            'arithmetic_operator': '-',
            'measurements': [('m1', 0., .1)]
        }

    def make_instance(self, identifier=None, registry=None):
        kwargs = self.make_kwargs()
        return self.class_to_test(identifier=identifier, **kwargs, registry=registry)

    def assert_equal_instance_except_id(self, lhs: ArithmeticAtomicPulseTemplate, rhs: ArithmeticAtomicPulseTemplate):
        self.assertIsInstance(lhs, ArithmeticAtomicPulseTemplate)
        self.assertIsInstance(rhs, ArithmeticAtomicPulseTemplate)
        self.assertEqual(lhs.lhs, rhs.lhs)
        self.assertEqual(lhs.arithmetic_operator, rhs.arithmetic_operator)
        self.assertEqual(lhs.rhs, rhs.rhs)
        self.assertEqual(lhs.measurement_declarations, rhs.measurement_declarations)


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

    def test_try_operation(self):
        raise NotImplementedError()

    def test_build_waveform(self):
        raise NotImplementedError()
