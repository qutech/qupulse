import unittest
import warnings
from unittest import mock

from qupulse.parameter_scope import Scope, DictScope
from qupulse.utils.types import FrozenDict

from qupulse._program._loop import Loop
from qupulse.expressions import Expression, ExpressionScalar
from qupulse.pulses import ConstantPT
from qupulse.pulses.repetition_pulse_template import RepetitionPulseTemplate,ParameterNotIntegerException
from qupulse.pulses.parameters import ParameterNotProvidedException, ParameterConstraintViolation, ParameterConstraint

from tests.pulses.sequencing_dummies import DummyPulseTemplate, DummyWaveform, MeasurementWindowTestCase
from tests.serialization_dummies import DummySerializer
from tests.serialization_tests import SerializableTests
from tests._program.transformation_tests import TransformationStub
from tests.pulses.pulse_template_tests import PulseTemplateStub, get_appending_internal_create_program


class RepetitionPulseTemplateTest(unittest.TestCase):

    def test_init(self) -> None:
        body = DummyPulseTemplate()
        repetition_count = 3
        t = RepetitionPulseTemplate(body, repetition_count)
        self.assertEqual(repetition_count, t.repetition_count)
        self.assertEqual(body, t.body)

        repetition_count = 'foo'
        t = RepetitionPulseTemplate(body, repetition_count)
        self.assertEqual(repetition_count, t.repetition_count)
        self.assertEqual(body, t.body)

        with self.assertRaises(ValueError):
            RepetitionPulseTemplate(body, Expression(-1))

        with self.assertWarnsRegex(UserWarning, '0 repetitions',
                                   msg='RepetitionPulseTemplate did not raise a warning for 0 repetitions on consruction.'):
            RepetitionPulseTemplate(body, 0)

    def test_parameter_names_and_declarations(self) -> None:
        body = DummyPulseTemplate()
        t = RepetitionPulseTemplate(body, 5)
        self.assertEqual(body.parameter_names, t.parameter_names)

        body.parameter_names_ = {'foo', 't', 'bar'}
        self.assertEqual(body.parameter_names, t.parameter_names)

    def test_parameter_names(self) -> None:
        for body in [DummyPulseTemplate(parameter_names={'foo', 'bar'}), ConstantPT(1.4, {'A': 'foo', 'B': 'bar'})]:
            t = RepetitionPulseTemplate(body, 5, parameter_constraints={'foo > hugo'}, measurements=[('meas', 'd', 0)])
            self.assertEqual({'foo', 'bar', 'hugo', 'd'}, t.parameter_names)

    def test_str(self) -> None:
        body = DummyPulseTemplate()
        t = RepetitionPulseTemplate(body, 9)
        self.assertIsInstance(str(t), str)
        t = RepetitionPulseTemplate(body, 'foo')
        self.assertIsInstance(str(t), str)

    def test_measurement_names(self):
        measurement_names = {'M'}
        body = DummyPulseTemplate(measurement_names=measurement_names)
        t = RepetitionPulseTemplate(body, 9)

        self.assertEqual(measurement_names, t.measurement_names)

        t = RepetitionPulseTemplate(body, 9, measurements=[('N', 1, 2)])
        self.assertEqual({'M', 'N'}, t.measurement_names)

    def test_duration(self):
        body = DummyPulseTemplate(duration='foo')
        t = RepetitionPulseTemplate(body, 'bar')

        self.assertEqual(t.duration, Expression('foo*bar'))

    def test_integral(self) -> None:
        dummy = DummyPulseTemplate(integrals={'A': ExpressionScalar('foo+2'), 'B': ExpressionScalar('k*3+x**2')})
        template = RepetitionPulseTemplate(dummy, 7)
        self.assertEqual({'A': Expression('7*(foo+2)'), 'B': Expression('7*(k*3+x**2)')}, template.integral)

        template = RepetitionPulseTemplate(dummy, '2+m')
        self.assertEqual({'A': Expression('(2+m)*(foo+2)'), 'B': Expression('(2+m)*(k*3+x**2)')}, template.integral)

        template = RepetitionPulseTemplate(dummy, Expression('2+m'))
        self.assertEqual({'A': Expression('(2+m)*(foo+2)'), 'B': Expression('(2+m)*(k*3+x**2)')}, template.integral)

    def test_initial_values(self):
        dummy = DummyPulseTemplate(initial_values={'A': ExpressionScalar('a + 3')})
        rpt = RepetitionPulseTemplate(dummy, repetition_count='n')
        self.assertEqual(dummy.initial_values, rpt.initial_values)

    def test_final_values(self):
        dummy = DummyPulseTemplate(final_values={'A': ExpressionScalar('a + 3')})
        rpt = RepetitionPulseTemplate(dummy, repetition_count='n')
        self.assertEqual(dummy.final_values, rpt.final_values)

    def test_parameter_names_param_only_in_constraint(self) -> None:
        pt = RepetitionPulseTemplate(DummyPulseTemplate(parameter_names={'a'}), 'n', parameter_constraints=['a<c'])
        self.assertEqual(pt.parameter_names, {'a','c', 'n'})


class RepetitionPulseTemplateSequencingTests(MeasurementWindowTestCase):
    def test_internal_create_program(self):
        wf = DummyWaveform(duration=2.)
        body = PulseTemplateStub()

        rpt = RepetitionPulseTemplate(body, 'n_rep*mul', measurements=[('m', 'a', 'b')])

        scope = DictScope.from_kwargs(n_rep=3,
                                      mul=2,
                                      a=.1,
                                      b=.2,
                                      irrelevant=42)
        measurement_mapping = {'m': 'l'}
        channel_mapping = {'x': 'Y'}
        global_transformation = TransformationStub()
        to_single_waveform = {'to', 'single', 'waveform'}

        program = Loop()
        expected_program = Loop(children=[Loop(children=[Loop(waveform=wf)], repetition_count=6)],
                                measurements=[('l', .1, .2)])

        real_relevant_parameters = dict(n_rep=3, mul=2, a=0.1, b=0.2)

        with mock.patch.object(body, '_create_program',
                               wraps=get_appending_internal_create_program(wf, always_append=True)) as body_create_program:
            with mock.patch.object(rpt, 'validate_scope') as validate_scope:
                with mock.patch.object(rpt, 'get_repetition_count_value', return_value=6) as get_repetition_count_value:
                    with mock.patch.object(rpt, 'get_measurement_windows', return_value=[('l', .1, .2)]) as get_meas:
                        rpt._internal_create_program(scope=scope,
                                                     measurement_mapping=measurement_mapping,
                                                     channel_mapping=channel_mapping,
                                                     global_transformation=global_transformation,
                                                     to_single_waveform=to_single_waveform,
                                                     parent_loop=program)

                        self.assertEqual(program, expected_program)
                        body_create_program.assert_called_once_with(scope=scope,
                                                                    measurement_mapping=measurement_mapping,
                                                                    channel_mapping=channel_mapping,
                                                                    global_transformation=global_transformation,
                                                                    to_single_waveform=to_single_waveform,
                                                                    parent_loop=program.children[0])
                        validate_scope.assert_called_once_with(scope)
                        get_repetition_count_value.assert_called_once_with(scope)
                        get_meas.assert_called_once_with(scope, measurement_mapping)

    def test_create_program_constant_success_measurements(self) -> None:
        repetitions = 3
        body = DummyPulseTemplate(duration=2.0, waveform=DummyWaveform(duration=2, defined_channels={'A'}), measurements=[('b', 0, 1)])
        t = RepetitionPulseTemplate(body, repetitions, parameter_constraints=['foo<9'], measurements=[('my', 2, 2)])
        scope = DictScope.from_mapping({'foo': 8})
        measurement_mapping = {'my': 'thy', 'b': 'b'}
        channel_mapping = {}
        program = Loop()
        t._internal_create_program(scope=scope,
                                   measurement_mapping=measurement_mapping,
                                   channel_mapping=channel_mapping,
                                   to_single_waveform=set(),
                                   global_transformation=None,
                                   parent_loop=program)

        self.assertEqual(1, len(program.children))
        internal_loop = program[0]  # type: Loop
        self.assertEqual(repetitions, internal_loop.repetition_count)

        self.assertEqual(1, len(internal_loop))
        self.assertEqual((scope, measurement_mapping, channel_mapping, internal_loop), body.create_program_calls[-1])
        self.assertEqual(body.waveform, internal_loop[0].waveform)

        self.assert_measurement_windows_equal({'b': ([0, 2, 4], [1, 1, 1]), 'thy': ([2], [2])}, program.get_measurement_windows())

        # done in MultiChannelProgram
        program.cleanup()

        self.assert_measurement_windows_equal({'b': ([0, 2, 4], [1, 1, 1]), 'thy': ([2], [2])},
                                              program.get_measurement_windows())

    def test_create_program_declaration_success(self) -> None:
        repetitions = "foo"
        body = DummyPulseTemplate(duration=2.0, waveform=DummyWaveform(duration=2, defined_channels={'A'}))
        t = RepetitionPulseTemplate(body, repetitions, parameter_constraints=['foo<9'])
        scope = DictScope.from_kwargs(foo=3)
        measurement_mapping = dict(moth='fire')
        channel_mapping = dict(asd='f')
        program = Loop()
        t._internal_create_program(scope=scope,
                                   measurement_mapping=measurement_mapping,
                                   channel_mapping=channel_mapping,
                                   to_single_waveform=set(),
                                   global_transformation=None,
                                   parent_loop=program)

        self.assertEqual(1, program.repetition_count)
        self.assertEqual(1, len(program.children))
        internal_loop = program.children[0]  # type: Loop
        self.assertEqual(scope[repetitions], internal_loop.repetition_count)

        self.assertEqual(1, len(internal_loop))
        self.assertEqual((scope, measurement_mapping, channel_mapping, internal_loop),
                         body.create_program_calls[-1])
        self.assertEqual(body.waveform, internal_loop[0].waveform)

        self.assert_measurement_windows_equal({}, program.get_measurement_windows())

        # ensure same result as from Sequencer
        ## not the same as from Sequencer. Sequencer simplifies the whole thing to a single loop executing the waveform 3 times
        ## due to absence of non-repeated measurements. create_program currently does no such optimization

    def test_create_program_declaration_success_appended_measurements(self) -> None:
        repetitions = "foo"
        body = DummyPulseTemplate(duration=2.0, waveform=DummyWaveform(duration=2), measurements=[('b', 0, 1)])
        t = RepetitionPulseTemplate(body, repetitions, parameter_constraints=['foo<9'],
                                    measurements=[('moth', 0, 'meas_end')])
        scope = DictScope.from_kwargs(foo=3, meas_end=7.1)
        measurement_mapping = dict(moth='fire', b='b')
        channel_mapping = dict(asd='f')
        children = [Loop(waveform=DummyWaveform(duration=0))]
        program = Loop(children=children, measurements=[('a', [0], [1])], repetition_count=2)

        t._internal_create_program(scope=scope,
                                   measurement_mapping=measurement_mapping,
                                   channel_mapping=channel_mapping,
                                   to_single_waveform=set(),
                                   global_transformation=None,
                                   parent_loop=program)

        self.assertEqual(2, program.repetition_count)
        self.assertEqual(2, len(program.children))
        self.assertIs(program.children[0], children[0])
        internal_loop = program.children[1]  # type: Loop
        self.assertEqual(scope[repetitions], internal_loop.repetition_count)

        self.assertEqual(1, len(internal_loop))
        self.assertEqual((scope, measurement_mapping, channel_mapping, internal_loop), body.create_program_calls[-1])
        self.assertEqual(body.waveform, internal_loop[0].waveform)

        self.assert_measurement_windows_equal({'fire': ([0, 6], [7.1, 7.1]),
                                         'b': ([0, 2, 4, 6, 8, 10], [1, 1, 1, 1, 1, 1]),
                                         'a': ([0], [1])}, program.get_measurement_windows())

        # not ensure same result as from Sequencer here - we're testing appending to an already existing parent loop
        # which is a use case that does not immediately arise from using Sequencer

    def test_create_program_declaration_success_measurements(self) -> None:
        repetitions = "foo"
        body = DummyPulseTemplate(duration=2.0, waveform=DummyWaveform(duration=2), measurements=[('b', 0, 1)])
        t = RepetitionPulseTemplate(body, repetitions, parameter_constraints=['foo<9'], measurements=[('moth', 0, 'meas_end')])
        scope = DictScope.from_kwargs(foo=3, meas_end=7.1)
        measurement_mapping = dict(moth='fire', b='b')
        channel_mapping = dict(asd='f')
        program = Loop()
        t._internal_create_program(scope=scope,
                                   measurement_mapping=measurement_mapping,
                                   channel_mapping=channel_mapping,
                                   to_single_waveform=set(),
                                   global_transformation=None,
                                   parent_loop=program)

        self.assertEqual(1, program.repetition_count)
        self.assertEqual(1, len(program.children))
        internal_loop = program.children[0]  # type: Loop
        self.assertEqual(scope[repetitions], internal_loop.repetition_count)

        self.assertEqual(1, len(internal_loop))
        self.assertEqual((scope, measurement_mapping, channel_mapping, internal_loop), body.create_program_calls[-1])
        self.assertEqual(body.waveform, internal_loop[0].waveform)

        self.assert_measurement_windows_equal({'fire': ([0], [7.1]), 'b': ([0, 2, 4], [1, 1, 1])}, program.get_measurement_windows())

    def test_create_program_declaration_exceeds_bounds(self) -> None:
        repetitions = "foo"
        body_program = Loop(waveform=DummyWaveform(duration=1.0))
        body = DummyPulseTemplate(duration=2.0, program=body_program)
        t = RepetitionPulseTemplate(body, repetitions, parameter_constraints=['foo<9'])
        scope = DictScope.from_kwargs(foo=9)
        measurement_mapping = dict(moth='fire')
        channel_mapping = dict(asd='f')

        children = [Loop(waveform=DummyWaveform(duration=0))]
        program = Loop(children=children)
        with self.assertRaises(ParameterConstraintViolation):
            t._internal_create_program(scope=scope,
                                       measurement_mapping=measurement_mapping,
                                       channel_mapping=channel_mapping,
                                       to_single_waveform=set(),
                                       global_transformation=None,
                                       parent_loop=program)
        self.assertFalse(body.create_program_calls)
        self.assertEqual(1, program.repetition_count)
        self.assertEqual(children, list(program.children))
        self.assertIsNone(program.waveform)
        self.assert_measurement_windows_equal({}, program.get_measurement_windows())

    def test_create_program_declaration_parameter_not_provided(self) -> None:
        repetitions = "foo"
        body = DummyPulseTemplate(waveform=DummyWaveform(duration=2.0))
        t = RepetitionPulseTemplate(body, repetitions, parameter_constraints=['foo<9'], measurements=[('a', 'd', 1)])
        scope = DictScope.from_kwargs()
        measurement_mapping = dict(moth='fire')
        channel_mapping = dict(asd='f')
        children = [Loop(waveform=DummyWaveform(duration=0))]
        program = Loop(children=children)
        with self.assertRaises(ParameterNotProvidedException):
            t._internal_create_program(scope=scope,
                                       measurement_mapping=measurement_mapping,
                                       channel_mapping=channel_mapping,
                                   to_single_waveform=set(),
                                   global_transformation=None,
                                       parent_loop=program)

        with self.assertRaises(ParameterNotProvidedException):
            t._internal_create_program(scope=scope,
                                       measurement_mapping=measurement_mapping,
                                       channel_mapping=channel_mapping,
                                   to_single_waveform=set(),
                                   global_transformation=None,
                                       parent_loop=program)

        self.assertFalse(body.create_program_calls)
        self.assertEqual(1, program.repetition_count)
        self.assertEqual(children, list(program.children))
        self.assertIsNone(program.waveform)
        self.assert_measurement_windows_equal({}, program.get_measurement_windows())

    def test_create_program_declaration_parameter_value_not_whole(self) -> None:
        repetitions = "foo"
        body = DummyPulseTemplate(duration=2.0, waveform=DummyWaveform(duration=2.0))
        t = RepetitionPulseTemplate(body, repetitions, parameter_constraints=['foo<9'])
        scope = DictScope.from_kwargs(foo=(3.3))
        measurement_mapping = dict(moth='fire')
        channel_mapping = dict(asd='f')
        children = [Loop(waveform=DummyWaveform(duration=0))]
        program = Loop(children=children)
        with self.assertRaises(ParameterNotIntegerException):
            t._internal_create_program(scope=scope,
                                       measurement_mapping=measurement_mapping,
                                       channel_mapping=channel_mapping,
                                   to_single_waveform=set(),
                                   global_transformation=None,
                                       parent_loop=program)
        self.assertFalse(body.create_program_calls)
        self.assertEqual(1, program.repetition_count)
        self.assertEqual(children, list(program.children))
        self.assertIsNone(program.waveform)
        self.assert_measurement_windows_equal({}, program.get_measurement_windows())

    def test_create_program_constant_measurement_mapping_failure(self) -> None:
        repetitions = "foo"
        body = DummyPulseTemplate(duration=2.0, waveform=DummyWaveform(duration=2.0), measurements=[('b', 0, 1)])
        t = RepetitionPulseTemplate(body, repetitions, parameter_constraints=['foo<9'], measurements=[('a', 0, 1)])
        scope = DictScope.from_kwargs(foo=3)
        measurement_mapping = dict()
        channel_mapping = dict(asd='f')
        children = [Loop(waveform=DummyWaveform(duration=0))]
        program = Loop(children=children)
        with self.assertRaises(KeyError):
            t._internal_create_program(scope=scope,
                                       measurement_mapping=measurement_mapping,
                                       channel_mapping=channel_mapping,
                                       to_single_waveform=set(),
                                       global_transformation=None,
                                       parent_loop=program)

        # test for failure on child level
        measurement_mapping = dict(a='a')
        with self.assertRaises(KeyError):
            t._internal_create_program(scope=scope,
                                       measurement_mapping=measurement_mapping,
                                       channel_mapping=channel_mapping,
                                       to_single_waveform=set(),
                                       global_transformation=None,
                                       parent_loop=program)
        self.assertFalse(body.create_program_calls)
        self.assertEqual(1, program.repetition_count)
        self.assertEqual(children, list(program.children))
        self.assertIsNone(program.waveform)
        self.assert_measurement_windows_equal({}, program.get_measurement_windows())

    def test_create_program_rep_count_zero_constant(self) -> None:
        repetitions = 0
        body_program = Loop(waveform=DummyWaveform(duration=1.0))
        body = DummyPulseTemplate(duration=2.0, program=body_program)

        # suppress warning about 0 repetitions on construction here, we are only interested in correct behavior during sequencing (i.e., do nothing)
        with warnings.catch_warnings(record=True):
            t = RepetitionPulseTemplate(body, repetitions)

        scope = DictScope.from_kwargs()
        measurement_mapping = dict(moth='fire')
        channel_mapping = dict(asd='f')

        program = Loop()
        t._internal_create_program(scope=scope,
                                   measurement_mapping=measurement_mapping,
                                   channel_mapping=channel_mapping,
                                   to_single_waveform=set(),
                                   global_transformation=None,
                                   parent_loop=program)
        self.assertFalse(body.create_program_calls)
        self.assertFalse(program.children)
        self.assertEqual(1, program.repetition_count)
        self.assertEqual(None, program._measurements)

    def test_create_program_rep_count_zero_constant_with_measurement(self) -> None:
        repetitions = 0
        body_program = Loop(waveform=DummyWaveform(duration=1.0))
        body = DummyPulseTemplate(duration=2.0, program=body_program)

        # suppress warning about 0 repetitions on construction here, we are only interested in correct behavior during sequencing (i.e., do nothing)
        with warnings.catch_warnings(record=True):
            t = RepetitionPulseTemplate(body, repetitions, measurements=[('moth', 0, 'meas_end')])

        scope = DictScope.from_kwargs(meas_end=7.1)
        measurement_mapping = dict(moth='fire')
        channel_mapping = dict(asd='f')

        program = Loop()
        t._internal_create_program(scope=scope,
                                   measurement_mapping=measurement_mapping,
                                   channel_mapping=channel_mapping,
                                   to_single_waveform=set(),
                                   global_transformation=None,
                                   parent_loop=program)
        self.assertFalse(body.create_program_calls)
        self.assertFalse(program.children)
        self.assertEqual(1, program.repetition_count)
        self.assertEqual(None, program._measurements)

    def test_create_program_rep_count_zero_declaration(self) -> None:
        repetitions = "foo"
        body_program = Loop(waveform=DummyWaveform(duration=1.0))
        body = DummyPulseTemplate(duration=2.0, program=body_program)

        # suppress warning about 0 repetitions on construction here, we are only interested in correct behavior during sequencing (i.e., do nothing)
        with warnings.catch_warnings(record=True):
            t = RepetitionPulseTemplate(body, repetitions)

        scope = DictScope.from_kwargs(foo=0)
        measurement_mapping = dict(moth='fire')
        channel_mapping = dict(asd='f')

        program = Loop()
        t._internal_create_program(scope=scope,
                                   measurement_mapping=measurement_mapping,
                                   channel_mapping=channel_mapping,
                                   to_single_waveform=set(),
                                   global_transformation=None,
                                   parent_loop=program)
        self.assertFalse(body.create_program_calls)
        self.assertFalse(program.children)
        self.assertEqual(1, program.repetition_count)
        self.assertEqual(None, program._measurements)

    def test_create_program_rep_count_zero_declaration_with_measurement(self) -> None:
        repetitions = "foo"
        body_program = Loop(waveform=DummyWaveform(duration=1.0))
        body = DummyPulseTemplate(duration=2.0, program=body_program)

        # suppress warning about 0 repetitions on construction here, we are only interested in correct behavior during sequencing (i.e., do nothing)
        with warnings.catch_warnings(record=True):
            t = RepetitionPulseTemplate(body, repetitions, measurements=[('moth', 0, 'meas_end')])

        scope = DictScope.from_kwargs(foo=0, meas_end=7.1)
        measurement_mapping = dict(moth='fire')
        channel_mapping = dict(asd='f')

        program = Loop()
        t._internal_create_program(scope=scope,
                                   measurement_mapping=measurement_mapping,
                                   channel_mapping=channel_mapping,
                                   to_single_waveform=set(),
                                   global_transformation=None,
                                   parent_loop=program)
        self.assertFalse(body.create_program_calls)
        self.assertFalse(program.children)
        self.assertEqual(1, program.repetition_count)
        self.assertEqual(None, program._measurements)

    def test_create_program_rep_count_neg_declaration(self) -> None:
        repetitions = "foo"
        body_program = Loop(waveform=DummyWaveform(duration=1.0))
        body = DummyPulseTemplate(duration=2.0, program=body_program)

        # suppress warning about 0 repetitions on construction here, we are only interested in correct behavior during sequencing (i.e., do nothing)
        with warnings.catch_warnings(record=True):
            t = RepetitionPulseTemplate(body, repetitions)

        scope = DictScope.from_kwargs(foo=-1)
        measurement_mapping = dict(moth='fire')
        channel_mapping = dict(asd='f')

        program = Loop()
        t._internal_create_program(scope=scope,
                                   measurement_mapping=measurement_mapping,
                                   channel_mapping=channel_mapping,
                                   to_single_waveform=set(),
                                   global_transformation=None,
                                   parent_loop=program)
        self.assertFalse(body.create_program_calls)
        self.assertFalse(program.children)
        self.assertEqual(1, program.repetition_count)
        self.assertEqual(None, program._measurements)

    def test_create_program_rep_count_neg_declaration_with_measurements(self) -> None:
        repetitions = "foo"
        body_program = Loop(waveform=DummyWaveform(duration=1.0))
        body = DummyPulseTemplate(duration=2.0, program=body_program)

        # suppress warning about 0 repetitions on construction here, we are only interested in correct behavior during sequencing (i.e., do nothing)
        with warnings.catch_warnings(record=True):
            t = RepetitionPulseTemplate(body, repetitions, measurements=[('moth', 0, 'meas_end')])

        scope = DictScope.from_kwargs(foo=-1, meas_end=7.1)
        measurement_mapping = dict(moth='fire')
        channel_mapping = dict(asd='f')

        program = Loop()
        t._internal_create_program(scope=scope,
                                   measurement_mapping=measurement_mapping,
                                   channel_mapping=channel_mapping,
                                   to_single_waveform=set(),
                                   global_transformation=None,
                                   parent_loop=program)
        self.assertFalse(body.create_program_calls)
        self.assertFalse(program.children)
        self.assertEqual(1, program.repetition_count)
        self.assertEqual(None, program._measurements)

    def test_create_program_none_subprogram(self) -> None:
        repetitions = "foo"
        body = DummyPulseTemplate(duration=0.0, waveform=None)
        t = RepetitionPulseTemplate(body, repetitions, parameter_constraints=['foo<9'])
        scope = DictScope.from_kwargs(foo=3)
        measurement_mapping = dict(moth='fire')
        channel_mapping = dict(asd='f')
        program = Loop()
        t._internal_create_program(scope=scope,
                                   measurement_mapping=measurement_mapping,
                                   channel_mapping=channel_mapping,
                                   to_single_waveform=set(),
                                   global_transformation=None,
                                   parent_loop=program)
        self.assertFalse(program.children)
        self.assertEqual(1, program.repetition_count)
        self.assertEqual(None, program._measurements)

    def test_create_program_none_subprogram_with_measurement(self) -> None:
        repetitions = "foo"
        body = DummyPulseTemplate(duration=2.0, waveform=None, measurements=[('b', 2, 3)])
        t = RepetitionPulseTemplate(body, repetitions, parameter_constraints=['foo<9'],
                                    measurements=[('moth', 0, 'meas_end')])
        scope = DictScope.from_kwargs(foo=3, meas_end=7.1)
        measurement_mapping = dict(moth='fire', b='b')
        channel_mapping = dict(asd='f')
        program = Loop()

        t._internal_create_program(scope=scope,
                                   measurement_mapping=measurement_mapping,
                                   channel_mapping=channel_mapping,
                                   to_single_waveform=set(),
                                   global_transformation=None,
                                   parent_loop=program)
        self.assertFalse(program.children)
        self.assertEqual(1, program.repetition_count)
        self.assertEqual(None, program._measurements)


class RepetitionPulseTemplateSerializationTests(SerializableTests, unittest.TestCase):

    @property
    def class_to_test(self):
        return RepetitionPulseTemplate

    def make_kwargs(self):
        return {
            'body': DummyPulseTemplate(),
            'repetition_count': 3,
            'parameter_constraints': [str(ParameterConstraint('a<b'))],
            'measurements': [('m', 0, 1)]
        }

    def assert_equal_instance_except_id(self, lhs: RepetitionPulseTemplate, rhs: RepetitionPulseTemplate):
        self.assertIsInstance(lhs, RepetitionPulseTemplate)
        self.assertIsInstance(rhs, RepetitionPulseTemplate)
        self.assertEqual(lhs.body, rhs.body)
        self.assertEqual(lhs.parameter_constraints, rhs.parameter_constraints)
        self.assertEqual(lhs.measurement_declarations, rhs.measurement_declarations)


class RepetitionPulseTemplateOldSerializationTests(unittest.TestCase):

    def test_get_serialization_data_minimal_old(self) -> None:
        # test for deprecated version during transition period, remove after final switch
        with self.assertWarnsRegex(DeprecationWarning, "deprecated",
                                   msg="RepetitionPT does not issue warning for old serialization routines."):
            serializer = DummySerializer(deserialize_callback=lambda x: x['name'])
            body = DummyPulseTemplate()
            repetition_count = 3
            template = RepetitionPulseTemplate(body, repetition_count)
            expected_data = dict(
                body=str(id(body)),
                repetition_count=repetition_count,
            )
            data = template.get_serialization_data(serializer)
            self.assertEqual(expected_data, data)

    def test_get_serialization_data_all_features_old(self) -> None:
        # test for deprecated version during transition period, remove after final switch
        with self.assertWarnsRegex(DeprecationWarning, "deprecated",
                                   msg="RepetitionPT does not issue warning for old serialization routines."):
            serializer = DummySerializer(deserialize_callback=lambda x: x['name'])
            body = DummyPulseTemplate()
            repetition_count = 'foo'
            measurements = [('a', 0, 1), ('b', 1, 1)]
            parameter_constraints = ['foo < 3']
            template = RepetitionPulseTemplate(body, repetition_count,
                                               measurements=measurements,
                                               parameter_constraints=parameter_constraints)
            expected_data = dict(
                body=str(id(body)),
                repetition_count=repetition_count,
                measurements=measurements,
                parameter_constraints=parameter_constraints
            )
            data = template.get_serialization_data(serializer)
            self.assertEqual(expected_data, data)

    def test_deserialize_minimal_old(self) -> None:
        # test for deprecated version during transition period, remove after final switch
        with self.assertWarnsRegex(DeprecationWarning, "deprecated",
                                   msg="RepetitionPT does not issue warning for old serialization routines."):
            serializer = DummySerializer(deserialize_callback=lambda x: x['name'])
            body = DummyPulseTemplate()
            repetition_count = 3
            data = dict(
                repetition_count=repetition_count,
                body=dict(name=str(id(body))),
                identifier='foo'
            )
            # prepare dependencies for deserialization
            serializer.subelements[str(id(body))] = body
            # deserialize
            template = RepetitionPulseTemplate.deserialize(serializer, **data)
            # compare!
            self.assertIs(body, template.body)
            self.assertEqual(repetition_count, template.repetition_count)
            #self.assertEqual([str(c) for c in template.parameter_constraints], ['bar < 3'])

    def test_deserialize_all_features_old(self) -> None:
        # test for deprecated version during transition period, remove after final switch
        with self.assertWarnsRegex(DeprecationWarning, "deprecated",
                                   msg="RepetitionPT does not issue warning for old serialization routines."):
            serializer = DummySerializer(deserialize_callback=lambda x: x['name'])
            body = DummyPulseTemplate()
            data = dict(
                repetition_count='foo',
                body=dict(name=str(id(body))),
                identifier='foo',
                parameter_constraints=['foo < 3'],
                measurements=[('a', 0, 1), ('b', 1, 1)]
            )
            # prepare dependencies for deserialization
            serializer.subelements[str(id(body))] = body

            # deserialize
            template = RepetitionPulseTemplate.deserialize(serializer, **data)

            # compare!
            self.assertIs(body, template.body)
            self.assertEqual('foo', template.repetition_count)
            self.assertEqual(template.parameter_constraints, [ParameterConstraint('foo < 3')])
            self.assertEqual(template.measurement_declarations, data['measurements'])


class ParameterNotIntegerExceptionTests(unittest.TestCase):

    def test(self) -> None:
        exception = ParameterNotIntegerException('foo', 3)
        self.assertIsInstance(str(exception), str)


if __name__ == "__main__":
    unittest.main(verbosity=2)