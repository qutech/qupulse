import unittest
from unittest import mock

import numpy as np

from qupulse.pulses.arithmetic_pulse_template import ArithmeticAtomicPulseTemplate, ArithmeticPulseTemplate
from qupulse._program.waveforms import ArithmeticWaveform

from tests.pulses.sequencing_dummies import DummyPulseTemplate, DummyWaveform


class ArithmeticAtomicPulseTemplateTest(unittest.TestCase):
    def test_init(self):
        raise NotImplementedError()
    
    def test_requires_stop(self):
        raise NotImplementedError()
    
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

    def test_try_opration(self):
        raise NotImplementedError()

    def test_build_waveform(self):
        raise NotImplementedError()
