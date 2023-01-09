import unittest
from unittest import mock
import warnings

import numpy as np
import sympy

from qupulse.parameter_scope import DictScope
from qupulse.expressions import ExpressionScalar
from qupulse.pulses import ConstantPT
from qupulse.plotting import render
from qupulse.pulses.arithmetic_pulse_template import ArithmeticAtomicPulseTemplate, ArithmeticPulseTemplate,\
    ImplicitAtomicityInArithmeticPT, UnequalDurationWarningInArithmeticPT, try_operation
from qupulse._program.waveforms import TransformingWaveform
from qupulse._program.transformation import OffsetTransformation, ScalingTransformation, IdentityTransformation

from tests.pulses.sequencing_dummies import DummyPulseTemplate, DummyWaveform
from tests.pulses.pulse_template_tests import PulseTemplateStub
from tests.serialization_tests import SerializableTests
from qupulse.pulses import TablePT, FunctionPT, RepetitionPT, AtomicMultiChannelPT


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
            with mock.patch('qupulse.pulses.arithmetic_pulse_template.ArithmeticWaveform.from_operator', return_value=wf_arith) as wf_init:
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

        lhs = DummyPulseTemplate(duration='t_dur', defined_channels={'a', 'b'},
                                 parameter_names={'x', 'y'}, integrals=integrals_lhs)
        rhs = DummyPulseTemplate(duration='t_dur', defined_channels={'a', 'c'},
                                 parameter_names={'x', 'z'}, integrals=integrals_rhs)

        expected_plus = dict(a=ExpressionScalar('a_lhs + a_rhs'),
                             b=ExpressionScalar('b'),
                             c=ExpressionScalar('c'))
        expected_minus = dict(a=ExpressionScalar('a_lhs - a_rhs'),
                              b=ExpressionScalar('b'),
                              c=ExpressionScalar('-c'))
        self.assertEqual(expected_plus, (lhs + rhs).integral)
        self.assertEqual(expected_minus, (lhs - rhs).integral)

    def test_initial_final_values(self):
        lhs = DummyPulseTemplate(initial_values={'A': .1, 'B': 'b*2'}, final_values={'A': .2, 'B': 'b / 2'})
        rhs = DummyPulseTemplate(initial_values={'A': -4, 'B': 'b*2 + 1'}, final_values={'A': .2, 'B': '-b / 2 + c'})

        minus = lhs - rhs
        plus = lhs + rhs
        self.assertEqual({'A': 4.1, 'B': -1}, minus.initial_values)
        self.assertEqual({'A': 0, 'B': 'b - c'}, minus.final_values)

        self.assertEqual({'A': -3.9, 'B': 'b*4 + 1'}, plus.initial_values)
        self.assertEqual({'A': .4, 'B': 'c'}, plus.final_values)

    def test_as_expression(self):
        integrals_lhs = dict(a=ExpressionScalar('a_lhs'), b=ExpressionScalar('b'))
        integrals_rhs = dict(a=ExpressionScalar('a_rhs'), c=ExpressionScalar('c'))

        duration = 4
        t = DummyPulseTemplate._AS_EXPRESSION_TIME
        expr_lhs = {ch: i * t / duration**2 * 2 for ch, i in integrals_lhs.items()}
        expr_rhs = {ch: i * t / duration**2 * 2 for ch, i in integrals_rhs.items()}

        lhs = DummyPulseTemplate(duration=duration, defined_channels={'a', 'b'},
                                 parameter_names={'x', 'y'}, integrals=integrals_lhs)
        rhs = DummyPulseTemplate(duration=duration, defined_channels={'a', 'c'},
                                 parameter_names={'x', 'z'}, integrals=integrals_rhs)

        expected_added = {
            'a': expr_lhs['a'] + expr_rhs['a'],
            'b': expr_lhs['b'],
            'c': expr_rhs['c']
        }
        added_expr = (lhs + rhs)._as_expression()
        self.assertEqual(expected_added, added_expr)

        subs_expr = (lhs - rhs)._as_expression()
        expected_subs = {
            'a': expr_lhs['a'] - expr_rhs['a'],
            'b': expr_lhs['b'],
            'c': -expr_rhs['c']
        }
        self.assertEqual(expected_subs, subs_expr)

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
        lhs = DummyPulseTemplate(duration=4, defined_channels={'a', 'b'}, parameter_names={'x', 'y'})
        rhs = DummyPulseTemplate(duration=4, defined_channels={'a', 'c'}, parameter_names={'x', 'z'})

        with self.assertRaisesRegex(TypeError, 'needs to be a pulse template'):
            ArithmeticPulseTemplate('a', '+', 'b')

        with self.assertRaisesRegex(TypeError, 'two PulseTemplates'):
            ArithmeticPulseTemplate(lhs, '+', rhs)

        with self.assertRaisesRegex(ValueError, r'Operands \(scalar, PulseTemplate\) require'):
            ArithmeticPulseTemplate(4, '/', rhs)

        with self.assertRaisesRegex(ValueError, r'Operands \(PulseTemplate, scalar\) require'):
            ArithmeticPulseTemplate(lhs, '%', 4)

        scalar = mock.Mock()
        non_pt = mock.Mock()

        with mock.patch.object(ArithmeticPulseTemplate, '_parse_operand',
                               return_value=scalar) as parse_operand:
            with mock.patch('qupulse.pulses.arithmetic_pulse_template._is_time_dependent', return_value=False):
                arith = ArithmeticPulseTemplate(lhs, '/', non_pt)
                parse_operand.assert_called_once_with(non_pt, lhs.defined_channels)
                self.assertEqual(lhs, arith.lhs)
                self.assertEqual(scalar, arith.rhs)
                self.assertEqual(lhs, arith._pulse_template)
                self.assertEqual(scalar, arith._scalar)
                self.assertEqual('/', arith._arithmetic_operator)

        with mock.patch.object(ArithmeticPulseTemplate, '_parse_operand',
                               return_value=scalar) as parse_operand:
            with mock.patch('qupulse.pulses.arithmetic_pulse_template._is_time_dependent', return_value=False):
                arith = ArithmeticPulseTemplate(non_pt, '-', rhs)
                parse_operand.assert_called_once_with(non_pt, rhs.defined_channels)
                self.assertEqual(scalar, arith.lhs)
                self.assertEqual(rhs, arith.rhs)
                self.assertEqual(rhs, arith._pulse_template)
                self.assertEqual(scalar, arith._scalar)
                self.assertEqual('-', arith._arithmetic_operator)

        with mock.patch.object(ArithmeticPulseTemplate, '_parse_operand',
                               return_value=scalar) as parse_operand:
            with mock.patch('qupulse.pulses.arithmetic_pulse_template._is_time_dependent', return_value=True):
                with self.assertRaises(TypeError):
                    ArithmeticPulseTemplate(non_pt, '-', RepetitionPT(rhs, 3))

    def test_parse_operand(self):
        operand = {'a': 3, 'b': 'x'}
        with self.assertRaises(ValueError):
            ArithmeticPulseTemplate._parse_operand(operand, {'a'})

        self.assertEqual(dict(a=ExpressionScalar(3), b=ExpressionScalar('x')),
                         ArithmeticPulseTemplate._parse_operand(operand, {'a', 'b', 'c'}))

        expr_op = ExpressionScalar(3)
        self.assertIs(expr_op, ArithmeticPulseTemplate._parse_operand(expr_op, {'a', 'b', 'c'}))

        self.assertEqual(ExpressionScalar('foo'),
                         ArithmeticPulseTemplate._parse_operand('foo', {'a', 'b', 'c'}))

    def test_get_scalar_value(self):
        lhs = 'x + y'
        rhs = DummyPulseTemplate(defined_channels={'u', 'v', 'w'})
        arith = ArithmeticPulseTemplate(lhs, '-', rhs)

        parameters = dict(x=3, y=5, z=8)
        channel_mapping = dict(u='a', v='b', w=None)
        expected = dict(a=8, b=8)
        self.assertEqual(expected, arith._get_scalar_value(parameters=parameters,
                                                           channel_mapping=channel_mapping))

        lhs = {'u': 1., 'w': 3.}
        arith = ArithmeticPulseTemplate(lhs, '-', rhs)
        expected = dict(a=1.)
        self.assertEqual(expected, arith._get_scalar_value(parameters=parameters,
                                                           channel_mapping=channel_mapping))

    def test_get_transformation(self):
        pulse_template = DummyPulseTemplate(defined_channels={'u', 'v', 'w'})
        scalar = dict(a=1., b=2.)
        neg_scalar = dict(a=-1., b=-2.)
        inv_scalar = dict(a=1., b=1/2.)
        neg_trafo = ScalingTransformation(dict(a=-1, b=-1))

        parameters = dict(x=3, y=5, z=8)
        channel_mapping = dict(u='a', v='b', w=None)

        # (PT + scalar)
        arith = ArithmeticPulseTemplate(pulse_template, '+', 'd')
        with mock.patch.object(arith, '_get_scalar_value', return_value=scalar.copy()) as get_scalar_value:
            trafo = arith._get_transformation(parameters=parameters, channel_mapping=channel_mapping)
            get_scalar_value.assert_called_once_with(parameters=parameters, channel_mapping=channel_mapping)

            expected_trafo = OffsetTransformation(scalar)
            self.assertEqual(expected_trafo, trafo)

        # (scalar + PT)
        arith = ArithmeticPulseTemplate('d', '+', pulse_template)
        with mock.patch.object(arith, '_get_scalar_value', return_value=scalar.copy()) as get_scalar_value:
            trafo = arith._get_transformation(parameters=parameters, channel_mapping=channel_mapping)
            get_scalar_value.assert_called_once_with(parameters=parameters, channel_mapping=channel_mapping)

            expected_trafo = OffsetTransformation(scalar)
            self.assertEqual(expected_trafo, trafo)

        # (PT - scalar)
        arith = ArithmeticPulseTemplate(pulse_template, '-', 'd')
        with mock.patch.object(arith, '_get_scalar_value', return_value=scalar.copy()) as get_scalar_value:
            trafo = arith._get_transformation(parameters=parameters, channel_mapping=channel_mapping)
            get_scalar_value.assert_called_once_with(parameters=parameters, channel_mapping=channel_mapping)

            expected_trafo = OffsetTransformation(neg_scalar)
            self.assertEqual(expected_trafo, trafo)

        # (scalar - PT)
        arith = ArithmeticPulseTemplate('d', '-', pulse_template)
        with mock.patch.object(arith, '_get_scalar_value', return_value=scalar.copy()) as get_scalar_value:
            trafo = arith._get_transformation(parameters=parameters, channel_mapping=channel_mapping)
            get_scalar_value.assert_called_once_with(parameters=parameters, channel_mapping=channel_mapping)

            expected_trafo = neg_trafo.chain(OffsetTransformation(scalar))
            self.assertEqual(expected_trafo, trafo)

        # (PT * scalar)
        arith = ArithmeticPulseTemplate(pulse_template, '*', 'd')
        with mock.patch.object(arith, '_get_scalar_value', return_value=scalar.copy()) as get_scalar_value:
            trafo = arith._get_transformation(parameters=parameters, channel_mapping=channel_mapping)
            get_scalar_value.assert_called_once_with(parameters=parameters, channel_mapping=channel_mapping)

            expected_trafo = ScalingTransformation(scalar)
            self.assertEqual(expected_trafo, trafo)

        # (scalar * PT)
        arith = ArithmeticPulseTemplate('d', '*', pulse_template)
        with mock.patch.object(arith, '_get_scalar_value', return_value=scalar.copy()) as get_scalar_value:
            trafo = arith._get_transformation(parameters=parameters, channel_mapping=channel_mapping)
            get_scalar_value.assert_called_once_with(parameters=parameters, channel_mapping=channel_mapping)

            expected_trafo = ScalingTransformation(scalar)
            self.assertEqual(expected_trafo, trafo)

        # (PT / scalar)
        arith = ArithmeticPulseTemplate(pulse_template, '/', 'd')
        with mock.patch.object(arith, '_get_scalar_value', return_value=scalar.copy()) as get_scalar_value:
            trafo = arith._get_transformation(parameters=parameters, channel_mapping=channel_mapping)
            get_scalar_value.assert_called_once_with(parameters=parameters, channel_mapping=channel_mapping)

            expected_trafo = ScalingTransformation(inv_scalar)
            self.assertEqual(expected_trafo, trafo)

    def test_time_dependent_expression(self):
        inner = FunctionPT('exp(-(t - t_duration/2)**2)', duration_expression='t_duration')
        inner_iq = AtomicMultiChannelPT((inner, {'default': 'I'}), (inner, {'default': 'Q'}))
        modulated = ArithmeticPulseTemplate(inner_iq, '*', {'I': 'sin(2*pi*f*t)', 'Q': 'cos(2*pi*f*t)'})
        program = modulated.create_program(parameters={'t_duration': 10, 'f': 1.})
        wf = program[0].waveform
        self.assertEqual(1, len(program))
        time = np.linspace(0, 10)

        sampled_i = wf.get_sampled('I', time)
        sampled_q = wf.get_sampled('Q', time)

        expected_sampled_i = np.sin(2*np.pi*time) * np.exp(-(time - 5)**2)
        expected_sampled_q = np.cos(2*np.pi*time) * np.exp(-(time - 5)**2)
        np.testing.assert_allclose(expected_sampled_i, sampled_i)
        np.testing.assert_allclose(expected_sampled_q, sampled_q)

    def test_time_dependent_global_expression(self):
        # test case stems from failing example
        gauss = FunctionPT('ampl * exp(-((t - t_gauss/2) / tau)**2)', 't_gauss', 'C')

        gauss_iq = AtomicMultiChannelPT(
            gauss.with_mapping({'C': 'I'}) * 'cos(omega * t)',
            gauss.with_mapping({'C': 'Q'}) * 'sin(omega * t)',
        )
        program = gauss_iq.create_program(parameters={
            'ampl': 1.,
            'tau': 10.,
            't_gauss': 50,
            'omega': .2,
        })
        wf = program[0].waveform
        self.assertEqual(1, len(program))

        time = np.linspace(0, 50)

        sampled_i = wf.get_sampled('I', time)
        sampled_q = wf.get_sampled('Q', time)
        expected_sampled_i = np.cos(0.2*time) * np.exp(-((time - 25)/10)**2)
        expected_sampled_q = np.sin(0.2*time) * np.exp(-((time - 25)/10)**2)
        np.testing.assert_allclose(expected_sampled_i, sampled_i)
        np.testing.assert_allclose(expected_sampled_q, sampled_q)

    def test_time_dependent_integral(self):
        gauss = FunctionPT('sin(f * t)', 't_gauss', 'C').with_mapping({'f': 'omega'})
        gauss_mod = (gauss * 'sin(omega * t)').with_mapping({'omega': .1})
        symbolic, = gauss_mod.integral.values()
        t_gauss = np.linspace(0., 60., num=1000)
        expected = 0.5*t_gauss - 5.0*np.sin(0.1*t_gauss)*np.cos(0.1*t_gauss)
        evaluated = symbolic.evaluate_in_scope({'t_gauss': t_gauss})
        np.testing.assert_allclose(expected, evaluated)

    def test_internal_create_program(self):
        lhs = 'x + y'
        rhs = DummyPulseTemplate(defined_channels={'u', 'v', 'w'})
        arith = ArithmeticPulseTemplate(lhs, '-', rhs)

        scope = DictScope.from_kwargs(x=3, y=5, z=8, volatile={'some_parameter'})
        channel_mapping = dict(u='a', v='b', w=None)
        measurement_mapping = dict(m1='m2')
        global_transformation = OffsetTransformation({'unrelated': 1.})
        to_single_waveform = {'something_else'}
        parent_loop = mock.Mock()

        expected_transformation = mock.Mock(spec=IdentityTransformation())

        inner_trafo = mock.Mock(spec=IdentityTransformation())
        inner_trafo.chain.return_value = expected_transformation

        with mock.patch.object(rhs, '_create_program') as inner_create_program:
            with mock.patch.object(arith, '_get_transformation', return_value=inner_trafo) as get_transformation:
                arith._internal_create_program(
                    scope=scope,
                    measurement_mapping=measurement_mapping,
                    channel_mapping=channel_mapping,
                    global_transformation=global_transformation,
                    to_single_waveform=to_single_waveform,
                    parent_loop=parent_loop
                )
                get_transformation.assert_called_once_with(parameters=scope, channel_mapping=channel_mapping)

            inner_trafo.chain.assert_called_once_with(global_transformation)
            inner_create_program.assert_called_once_with(
                scope=scope,
                measurement_mapping=measurement_mapping,
                channel_mapping=channel_mapping,
                global_transformation=expected_transformation,
                to_single_waveform=to_single_waveform,
                parent_loop=parent_loop
            )

            with self.assertRaisesRegex(NotImplementedError, 'volatile'):
                arith._internal_create_program(
                    scope=DictScope.from_kwargs(x=3, y=5, z=8, volatile={'x'}),
                    measurement_mapping=measurement_mapping,
                    channel_mapping=channel_mapping,
                    global_transformation=global_transformation,
                    to_single_waveform=to_single_waveform,
                    parent_loop=parent_loop
                )

    def test_integral(self):
        scalar = 'x + y'
        mapping = {'u': 'x + y', 'v': 2.2}
        pt = DummyPulseTemplate(defined_channels={'u', 'v', 'w'}, integrals={'u': ExpressionScalar('ui'),
                                                                             'v': ExpressionScalar('vi'),
                                                                             'w': ExpressionScalar('wi')},
                                duration='t_dur')

        # commutative (+ scalar pt)
        expected = dict(u=ExpressionScalar('ui + (x + y) * t_dur'),
                        v=ExpressionScalar('vi + (x + y) * t_dur'),
                        w=ExpressionScalar('wi + (x + y) * t_dur'))
        self.assertEqual(expected, ArithmeticPulseTemplate(scalar, '+', pt).integral)
        self.assertEqual(expected, ArithmeticPulseTemplate(pt, '+', scalar).integral)

        # commutative (+ mapping pt)
        expected = dict(u=ExpressionScalar('ui + (x + y) * t_dur'),
                        v=ExpressionScalar('vi + 2.2 * t_dur'),
                        w=ExpressionScalar('wi'))
        self.assertEqual(expected, ArithmeticPulseTemplate(mapping, '+', pt).integral)
        self.assertEqual(expected, ArithmeticPulseTemplate(pt, '+', mapping).integral)

        # commutative (* scalar pt)
        expected = dict(u=ExpressionScalar('ui * (x + y)'),
                        v=ExpressionScalar('vi * (x + y)'),
                        w=ExpressionScalar('wi * (x + y)'))
        self.assertEqual(expected, ArithmeticPulseTemplate(scalar, '*', pt).integral)
        self.assertEqual(expected, ArithmeticPulseTemplate(pt, '*', scalar).integral)

        # commutative (* mapping pt)
        expected = dict(u=ExpressionScalar('ui * (x + y)'),
                        v=ExpressionScalar('vi * 2.2'),
                        w=ExpressionScalar('wi'))
        self.assertEqual(expected, ArithmeticPulseTemplate(mapping, '*', pt).integral)
        self.assertEqual(expected, ArithmeticPulseTemplate(pt, '*', mapping).integral)

        # (pt - scalar)
        expected = dict(u=ExpressionScalar('ui - (x + y) * t_dur'),
                        v=ExpressionScalar('vi - (x + y) * t_dur'),
                        w=ExpressionScalar('wi - (x + y) * t_dur'))
        self.assertEqual(expected, ArithmeticPulseTemplate(pt, '-', scalar).integral)

        # (scalar - pt)
        expected = dict(u=ExpressionScalar('(x + y) * t_dur - ui'),
                        v=ExpressionScalar('(x + y) * t_dur - vi'),
                        w=ExpressionScalar('(x + y) * t_dur - wi'))
        self.assertEqual(expected, ArithmeticPulseTemplate(scalar, '-', pt).integral)

        # (mapping - pt)
        expected = dict(u=ExpressionScalar('(x + y) * t_dur - ui'),
                        v=ExpressionScalar('2.2 * t_dur - vi'),
                        w=ExpressionScalar('-wi'))
        self.assertEqual(expected, ArithmeticPulseTemplate(mapping, '-', pt).integral)

        # (pt - mapping)
        expected = dict(u=ExpressionScalar('ui - (x + y) * t_dur'),
                        v=ExpressionScalar('vi - 2.2 * t_dur'),
                        w=ExpressionScalar('wi'))
        self.assertEqual(expected, ArithmeticPulseTemplate(pt, '-', mapping).integral)

        # (pt / scalar)
        expected = dict(u=ExpressionScalar('ui / (x + y)'),
                        v=ExpressionScalar('vi / (x + y)'),
                        w=ExpressionScalar('wi / (x + y)'))
        self.assertEqual(expected, ArithmeticPulseTemplate(pt, '/', scalar).integral)

        # (pt / mapping)
        expected = dict(u=ExpressionScalar('ui / (x + y)'),
                        v=ExpressionScalar('vi / 2.2'),
                        w=ExpressionScalar('wi'))
        self.assertEqual(expected, ArithmeticPulseTemplate(pt, '/', mapping).integral)

    def test_initial_values(self):
        lhs = DummyPulseTemplate(initial_values={'A': .3, 'B': 'b'}, defined_channels={'A', 'B'})
        apt = lhs + 'a'
        self.assertEqual({'A': 'a + 0.3', 'B': 'b + a'}, apt.initial_values)

    def test_final_values(self):
        lhs = DummyPulseTemplate(final_values={'A': .3, 'B': 'b'}, defined_channels={'A', 'B'})
        apt = lhs - 'a'
        self.assertEqual({'A': '-a + .3', 'B': 'b - a'}, apt.final_values)

    def test_simple_attributes(self):
        lhs = DummyPulseTemplate(defined_channels={'a', 'b'}, duration=ExpressionScalar('t_dur'),
                                 measurement_names={'m1'})
        rhs = 4
        arith = ArithmeticPulseTemplate(lhs, '+', rhs)
        self.assertIs(lhs.duration, arith.duration)
        self.assertIs(lhs.measurement_names, arith.measurement_names)

    def test_parameter_names(self):
        pt = DummyPulseTemplate(defined_channels={'a'}, parameter_names={'foo', 'bar'})
        scalar = 'x + y'

        arith = ArithmeticPulseTemplate(pt, '+', scalar)
        self.assertEqual(frozenset({'x', 'y'}), arith._scalar_operand_parameters)
        self.assertEqual({'x', 'y', 'foo', 'bar'}, arith.parameter_names)

        pt = DummyPulseTemplate(defined_channels={'a'}, parameter_names={'foo', 'bar'})
        self.assertEqual(frozenset({'x', 'y'}), arith._scalar_operand_parameters)
        self.assertEqual({'x', 'y', 'foo', 'bar'}, arith.parameter_names)

        arith = ArithmeticPulseTemplate(pt, '+', scalar + '+t')
        self.assertEqual({'x', 'y', 'foo', 'bar'}, arith.parameter_names)

    def test_try_operation(self):
        apt = DummyPulseTemplate(duration=1, defined_channels={'a'})
        npt = PulseTemplateStub(defined_channels={'a'})

        self.assertIsInstance(try_operation(npt, '+', 6), ArithmeticPulseTemplate)
        self.assertIsInstance(try_operation(apt, '+', apt), ArithmeticAtomicPulseTemplate)
        self.assertIs(NotImplemented, try_operation(npt, '/', npt))
        self.assertIs(NotImplemented, try_operation(npt, '//', 6))

    def test_build_waveform(self):
        pt = DummyPulseTemplate(defined_channels={'a'}, duration=6)

        parameters = dict(x=5., y=5.7)
        channel_mapping = dict(a='u', b='v')

        inner_wf = DummyWaveform(duration=6, defined_channels={'a'})
        trafo = mock.Mock(spec=IdentityTransformation())

        arith = ArithmeticPulseTemplate(pt, '-', 6)

        with mock.patch.object(pt, 'build_waveform', return_value=None) as inner_build:
            with mock.patch.object(arith, '_get_transformation') as _get_transformation:
                self.assertIsNone(arith.build_waveform(parameters=parameters, channel_mapping=channel_mapping))
                inner_build.assert_called_once_with(parameters=parameters, channel_mapping=channel_mapping)
                _get_transformation.assert_not_called()

        expected = TransformingWaveform(inner_wf, trafo)

        with mock.patch.object(pt, 'build_waveform', return_value=inner_wf) as inner_build:
            with mock.patch.object(arith, '_get_transformation', return_value=trafo) as _get_transformation:
                result = arith.build_waveform(parameters=parameters, channel_mapping=channel_mapping)
                inner_build.assert_called_once_with(parameters=parameters, channel_mapping=channel_mapping)
                _get_transformation.assert_called_once_with(parameters=parameters, channel_mapping=channel_mapping)
        self.assertEqual(expected, result)

    def test_repr(self):
        pt = DummyPulseTemplate(defined_channels={'a'})
        scalar = 'x'

        with mock.patch.object(DummyPulseTemplate, '__repr__', wraps=lambda *args: 'dummy'):
            r = repr(ArithmeticPulseTemplate(pt, '-', scalar))
        self.assertEqual("(dummy - ExpressionScalar('x'))", r)

        arith = ArithmeticPulseTemplate(pt, '-', scalar, identifier='id')
        self.assertEqual(super(ArithmeticPulseTemplate, arith).__repr__(), repr(arith))

    def test_time_dependence(self):
        inner = ConstantPT(1.4, {'a': ExpressionScalar('x'), 'b': 1.1})
        with self.assertRaises(TypeError):
            ArithmeticPulseTemplate(RepetitionPT(inner, 3), '*', {'a': 'sin(t)', 'b': 'cos(t)'})

        pc = ArithmeticPulseTemplate(inner, '*', {'a': 'sin(t)', 'b': 'cos(t)'})
        prog = pc.create_program(parameters={'x': -1})
        t, vals, _ = render(prog, sample_rate=10)
        expected_values = {
            'a': -np.sin(t),
            'b': 1.1 * np.cos(t)
        }
        np.testing.assert_equal(expected_values, vals)


class ArithmeticUsageTests(unittest.TestCase):
    def setUp(self) -> None:
        # define some building blocks
        self.sin_pt = FunctionPT('sin(omega*t)', 't_duration', channel='X')
        self.cos_pt = FunctionPT('sin(omega*t)', 't_duration', channel='Y')
        self.exp_pt = AtomicMultiChannelPT(self.sin_pt, self.cos_pt)
        self.tpt = TablePT({'X': [(0, 0), (3., 4.), ('t_duration', 2., 'linear')],
                            'Y': [(0, 1.), ('t_y', 5.), ('t_duration', 0., 'linear')]})
        self.complex_pt = RepetitionPT(self.tpt, 5) @ self.exp_pt
        self.parameters = dict(t_duration=10, omega=3.14*2/10, t_y=3.4)

    def test_scaling(self):
        from qupulse import plotting

        parameters = {**self.parameters, 'foo': 5.3}
        t_ref, reference, _ = plotting.render(self.complex_pt.create_program(parameters=parameters))

        for factor in (5, 5.3, 'foo'):
            scaled = factor * self.complex_pt
            real_scale = ExpressionScalar(factor).evaluate_numeric(**parameters)
            program = scaled.create_program(parameters=parameters)

            t, rendered, _ = plotting.render(program, 10.)
            np.testing.assert_equal(t_ref, t)
            for ch, volts in rendered.items():
                np.testing.assert_allclose(reference[ch] * real_scale, volts)

            divided = self.complex_pt / factor
            t, rendered, _ = plotting.render(divided.create_program(parameters=parameters), 10.)
            np.testing.assert_equal(t_ref, t)
            for ch, volts in rendered.items():
                np.testing.assert_allclose(reference[ch] / real_scale, volts)

            sel_scaled = {'X': factor} * self.complex_pt
            t, rendered, _ = plotting.render(sel_scaled.create_program(parameters=parameters), 10.)
            np.testing.assert_equal(t_ref, t)
            for ch, volts in rendered.items():
                if ch == 'X':
                    np.testing.assert_allclose(reference[ch] * real_scale, volts)
                else:
                    np.testing.assert_equal(reference[ch], volts)

    def test_offset(self):
        _ = 5.3 + self.complex_pt
        _ = 5 + self.complex_pt
        _ = 'foo' + self.complex_pt

        _ = self.complex_pt + 4.5
        _ = self.complex_pt - '4.5'


