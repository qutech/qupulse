import unittest
import warnings
from unittest import mock

from qctoolkit._program._loop import Loop, MultiChannelProgram
from qctoolkit.expressions import Expression
from qctoolkit.pulses.repetition_pulse_template import RepetitionPulseTemplate,ParameterNotIntegerException, RepetitionWaveform
from qctoolkit.pulses.parameters import ParameterNotProvidedException, ParameterConstraintViolation, ConstantParameter, \
    ParameterConstraint
from qctoolkit._program.instructions import REPJInstruction, InstructionPointer

from qctoolkit.pulses.sequencing import Sequencer

from tests.pulses.sequencing_dummies import DummyPulseTemplate, DummySequencer, DummyInstructionBlock, DummyParameter,\
    DummyCondition, DummyWaveform, MeasurementWindowTestCase
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
        body = DummyPulseTemplate(parameter_names={'foo', 'bar'})
        t = RepetitionPulseTemplate(body, 5, parameter_constraints={'foo > hugo'}, measurements=[('meas', 'd', 0)])

        self.assertEqual({'foo', 'bar', 'hugo', 'd'}, t.parameter_names)

    @unittest.skip('is interruptable not implemented for loops')
    def test_is_interruptable(self) -> None:
        body = DummyPulseTemplate(is_interruptable=False)
        t = RepetitionPulseTemplate(body, 6)
        self.assertFalse(t.is_interruptable)

        body.is_interruptable_ = True
        self.assertTrue(t.is_interruptable)

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
        dummy = DummyPulseTemplate(integrals=['foo+2', 'k*3+x**2'])
        template = RepetitionPulseTemplate(dummy, 7)
        self.assertEqual([Expression('7*(foo+2)'), Expression('7*(k*3+x**2)')], template.integral)

        template = RepetitionPulseTemplate(dummy, '2+m')
        self.assertEqual([Expression('(2+m)*(foo+2)'), Expression('(2+m)*(k*3+x**2)')], template.integral)

        template = RepetitionPulseTemplate(dummy, Expression('2+m'))
        self.assertEqual([Expression('(2+m)*(foo+2)'), Expression('(2+m)*(k*3+x**2)')], template.integral)

    def test_parameter_names_param_only_in_constraint(self) -> None:
        pt = RepetitionPulseTemplate(DummyPulseTemplate(parameter_names={'a'}), 'n', parameter_constraints=['a<c'])
        self.assertEqual(pt.parameter_names, {'a','c', 'n'})


class RepetitionPulseTemplateSequencingTests(MeasurementWindowTestCase):
    def test_internal_create_program(self):
        wf = DummyWaveform(duration=2.)
        body = PulseTemplateStub()

        rpt = RepetitionPulseTemplate(body, 'n_rep*mul', measurements=[('m', 'a', 'b')])

        parameters = dict(n_rep=ConstantParameter(3),
                          mul=ConstantParameter(2),
                          a=ConstantParameter(0.1),
                          b=ConstantParameter(0.2),
                          irrelevant=ConstantParameter(42))
        measurement_mapping = {'m': 'l'}
        channel_mapping = {'x': 'Y'}
        global_transformation = TransformationStub()
        to_single_waveform = {'to', 'single', 'waveform'}

        program = Loop()
        expected_program = Loop(children=[Loop(children=[Loop(waveform=wf)], repetition_count=6)],
                                measurements=[('l', .1, .2)])

        real_relevant_parameters = dict(n_rep=3, mul=2, a=0.1, b=0.2)

        with mock.patch.object(body, '_create_program',
                               wraps=get_appending_internal_create_program(wf, True)) as body_create_program:
            with mock.patch.object(rpt, 'validate_parameter_constraints') as validate_parameter_constraints:
                with mock.patch.object(rpt, 'get_repetition_count_value', return_value=6) as get_repetition_count_value:
                    with mock.patch.object(rpt, 'get_measurement_windows', return_value=[('l', .1, .2)]) as get_meas:
                        rpt._internal_create_program(parameters=parameters,
                                                     measurement_mapping=measurement_mapping,
                                                     channel_mapping=channel_mapping,
                                                     global_transformation=global_transformation,
                                                     to_single_waveform=to_single_waveform,
                                                     parent_loop=program)

                        self.assertEqual(program, expected_program)
                        body_create_program.assert_called_once_with(parameters=parameters,
                                                                    measurement_mapping=measurement_mapping,
                                                                    channel_mapping=channel_mapping,
                                                                    global_transformation=global_transformation,
                                                                    to_single_waveform=to_single_waveform,
                                                                    parent_loop=program.children[0])
                        validate_parameter_constraints.assert_called_once_with(parameters=parameters)
                        get_repetition_count_value.assert_called_once_with(real_relevant_parameters)
                        get_meas.assert_called_once_with(real_relevant_parameters, measurement_mapping)

    def test_create_program_constant_success_measurements(self) -> None:
        repetitions = 3
        body = DummyPulseTemplate(duration=2.0, waveform=DummyWaveform(duration=2, defined_channels={'A'}), measurements=[('b', 0, 1)])
        t = RepetitionPulseTemplate(body, repetitions, parameter_constraints=['foo<9'], measurements=[('my', 2, 2)])
        parameters = {'foo': 8}
        measurement_mapping = {'my': 'thy', 'b': 'b'}
        channel_mapping = {}
        program = Loop()
        t._internal_create_program(parameters=parameters,
                                   measurement_mapping=measurement_mapping,
                                   channel_mapping=channel_mapping,
                                   to_single_waveform=set(),
                                   global_transformation=None,
                                   parent_loop=program)

        self.assertEqual(1, len(program.children))
        internal_loop = program.children[0] # type: Loop
        self.assertEqual(repetitions, internal_loop.repetition_count)

        self.assertEqual(1, len(internal_loop))
        self.assertEqual((parameters, measurement_mapping, channel_mapping, internal_loop), body.create_program_calls[-1])
        self.assertEqual(body.waveform, internal_loop[0].waveform)

        self.assert_measurement_windows_equal({'b': ([0, 2, 4], [1, 1, 1]), 'thy': ([2], [2])}, program.get_measurement_windows())

        # ensure same result as from Sequencer
        sequencer = Sequencer()
        sequencer.push(t, parameters=parameters, conditions={}, window_mapping=measurement_mapping, channel_mapping=channel_mapping)
        block = sequencer.build()
        program_old = MultiChannelProgram(block, channels={'A'}).programs[frozenset({'A'})]
        self.assertEqual(program_old, program)

    def test_create_program_declaration_success(self) -> None:
        repetitions = "foo"
        body = DummyPulseTemplate(duration=2.0, waveform=DummyWaveform(duration=2, defined_channels={'A'}))
        t = RepetitionPulseTemplate(body, repetitions, parameter_constraints=['foo<9'])
        parameters = dict(foo=ConstantParameter(3))
        measurement_mapping = dict(moth='fire')
        channel_mapping = dict(asd='f')
        program = Loop()
        t._internal_create_program(parameters=parameters,
                                   measurement_mapping=measurement_mapping,
                                   channel_mapping=channel_mapping,
                                   to_single_waveform=set(),
                                   global_transformation=None,
                                   parent_loop=program)

        self.assertEqual(1, program.repetition_count)
        self.assertEqual(1, len(program.children))
        internal_loop = program.children[0]  # type: Loop
        self.assertEqual(parameters[repetitions].get_value(), internal_loop.repetition_count)

        self.assertEqual(1, len(internal_loop))
        self.assertEqual((parameters, measurement_mapping, channel_mapping, internal_loop), body.create_program_calls[-1])
        self.assertEqual(body.waveform, internal_loop[0].waveform)

        self.assert_measurement_windows_equal({}, program.get_measurement_windows())

        # ensure same result as from Sequencer
        ## not the same as from Sequencer. Sequencer simplifies the whole thing to a single loop executing the waveform 3 times
        ## due to absence of non-repeated measurements. create_program currently does no such optimization

    def test_create_program_declaration_success_appended_measurements(self) -> None:
        repetitions = "foo"
        body = DummyPulseTemplate(duration=2.0, waveform=DummyWaveform(duration=2), measurements=[('b', 0, 1)])
        t = RepetitionPulseTemplate(body, repetitions, parameter_constraints=['foo<9'], measurements=[('moth', 0, 'meas_end')])
        parameters = dict(foo=ConstantParameter(3), meas_end=ConstantParameter(7.1))
        measurement_mapping = dict(moth='fire', b='b')
        channel_mapping = dict(asd='f')

        children = [Loop(waveform=DummyWaveform(duration=0))]
        program = Loop(children=children, measurements=[('a', [0], [1])], repetition_count=2)

        t._internal_create_program(parameters=parameters,
                                   measurement_mapping=measurement_mapping,
                                   channel_mapping=channel_mapping,
                                   to_single_waveform=set(),
                                   global_transformation=None,
                                   parent_loop=program)

        self.assertEqual(2, program.repetition_count)
        self.assertEqual(2, len(program.children))
        self.assertIs(program.children[0], children[0])
        internal_loop = program.children[1]  # type: Loop
        self.assertEqual(parameters[repetitions].get_value(), internal_loop.repetition_count)

        self.assertEqual(1, len(internal_loop))
        self.assertEqual((parameters, measurement_mapping, channel_mapping, internal_loop), body.create_program_calls[-1])
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
        parameters = dict(foo=ConstantParameter(3), meas_end=ConstantParameter(7.1))
        measurement_mapping = dict(moth='fire', b='b')
        channel_mapping = dict(asd='f')
        program = Loop()
        t._internal_create_program(parameters=parameters,
                                   measurement_mapping=measurement_mapping,
                                   channel_mapping=channel_mapping,
                                   to_single_waveform=set(),
                                   global_transformation=None,
                                   parent_loop=program)

        self.assertEqual(1, program.repetition_count)
        self.assertEqual(1, len(program.children))
        internal_loop = program.children[0]  # type: Loop
        self.assertEqual(parameters[repetitions].get_value(), internal_loop.repetition_count)

        self.assertEqual(1, len(internal_loop))
        self.assertEqual((parameters, measurement_mapping, channel_mapping, internal_loop), body.create_program_calls[-1])
        self.assertEqual(body.waveform, internal_loop[0].waveform)

        self.assert_measurement_windows_equal({'fire': ([0], [7.1]), 'b': ([0, 2, 4], [1, 1, 1])}, program.get_measurement_windows())

        # ensure same result as from Sequencer
        sequencer = Sequencer()
        sequencer.push(t, parameters=parameters, conditions={}, window_mapping=measurement_mapping,
                       channel_mapping=channel_mapping)
        block = sequencer.build()
        program_old = MultiChannelProgram(block, channels={'A'}).programs[frozenset({'A'})]
        self.assertEqual(program_old, program)

    def test_create_program_declaration_exceeds_bounds(self) -> None:
        repetitions = "foo"
        body_program = Loop(waveform=DummyWaveform(duration=1.0))
        body = DummyPulseTemplate(duration=2.0, program=body_program)
        t = RepetitionPulseTemplate(body, repetitions, parameter_constraints=['foo<9'])
        parameters = dict(foo=ConstantParameter(9))
        measurement_mapping = dict(moth='fire')
        channel_mapping = dict(asd='f')

        children = [Loop(waveform=DummyWaveform(duration=0))]
        program = Loop(children=children)
        with self.assertRaises(ParameterConstraintViolation):
            t._internal_create_program(parameters=parameters,
                                       measurement_mapping=measurement_mapping,
                                       channel_mapping=channel_mapping,
                                   to_single_waveform=set(),
                                   global_transformation=None,
                                       parent_loop=program)
        self.assertFalse(body.create_program_calls)
        self.assertEqual(1, program.repetition_count)
        self.assertEqual(children, program.children)
        self.assertIsNone(program.waveform)
        self.assert_measurement_windows_equal({}, program.get_measurement_windows())

    def test_create_program_declaration_parameter_not_provided(self) -> None:
        repetitions = "foo"
        body = DummyPulseTemplate(waveform=DummyWaveform(duration=2.0))
        t = RepetitionPulseTemplate(body, repetitions, parameter_constraints=['foo<9'], measurements=[('a', 'd', 1)])
        parameters = {}
        measurement_mapping = dict(moth='fire')
        channel_mapping = dict(asd='f')
        children = [Loop(waveform=DummyWaveform(duration=0))]
        program = Loop(children=children)
        with self.assertRaises(ParameterNotProvidedException):
            t._internal_create_program(parameters=parameters,
                                       measurement_mapping=measurement_mapping,
                                       channel_mapping=channel_mapping,
                                   to_single_waveform=set(),
                                   global_transformation=None,
                                       parent_loop=program)

        parameters = {'foo': ConstantParameter(7)}
        with self.assertRaises(ParameterNotProvidedException):
            t._internal_create_program(parameters=parameters,
                                       measurement_mapping=measurement_mapping,
                                       channel_mapping=channel_mapping,
                                   to_single_waveform=set(),
                                   global_transformation=None,
                                       parent_loop=program)

        self.assertFalse(body.create_program_calls)
        self.assertEqual(1, program.repetition_count)
        self.assertEqual(children, program.children)
        self.assertIsNone(program.waveform)
        self.assert_measurement_windows_equal({}, program.get_measurement_windows())

    def test_create_program_declaration_parameter_value_not_whole(self) -> None:
        repetitions = "foo"
        body = DummyPulseTemplate(duration=2.0, waveform=DummyWaveform(duration=2.0))
        t = RepetitionPulseTemplate(body, repetitions, parameter_constraints=['foo<9'])
        parameters = dict(foo=ConstantParameter(3.3))
        measurement_mapping = dict(moth='fire')
        channel_mapping = dict(asd='f')
        children = [Loop(waveform=DummyWaveform(duration=0))]
        program = Loop(children=children)
        with self.assertRaises(ParameterNotIntegerException):
            t._internal_create_program(parameters=parameters,
                                       measurement_mapping=measurement_mapping,
                                       channel_mapping=channel_mapping,
                                   to_single_waveform=set(),
                                   global_transformation=None,
                                       parent_loop=program)
        self.assertFalse(body.create_program_calls)
        self.assertEqual(1, program.repetition_count)
        self.assertEqual(children, program.children)
        self.assertIsNone(program.waveform)
        self.assert_measurement_windows_equal({}, program.get_measurement_windows())

    def test_create_program_constant_measurement_mapping_failure(self) -> None:
        repetitions = "foo"
        body = DummyPulseTemplate(duration=2.0, waveform=DummyWaveform(duration=2.0), measurements=[('b', 0, 1)])
        t = RepetitionPulseTemplate(body, repetitions, parameter_constraints=['foo<9'], measurements=[('a', 0, 1)])
        parameters = dict(foo=ConstantParameter(3))
        measurement_mapping = dict()
        channel_mapping = dict(asd='f')
        children = [Loop(waveform=DummyWaveform(duration=0))]
        program = Loop(children=children)
        with self.assertRaises(KeyError):
            t._internal_create_program(parameters=parameters,
                                       measurement_mapping=measurement_mapping,
                                       channel_mapping=channel_mapping,
                                   to_single_waveform=set(),
                                   global_transformation=None,
                                       parent_loop=program)

        # test for failure on child level
        measurement_mapping = dict(a='a')
        with self.assertRaises(KeyError):
            t._internal_create_program(parameters=parameters,
                                       measurement_mapping=measurement_mapping,
                                       channel_mapping=channel_mapping,
                                   to_single_waveform=set(),
                                   global_transformation=None,
                                       parent_loop=program)
        self.assertFalse(body.create_program_calls)
        self.assertEqual(1, program.repetition_count)
        self.assertEqual(children, program.children)
        self.assertIsNone(program.waveform)
        self.assert_measurement_windows_equal({}, program.get_measurement_windows())

    def test_create_program_rep_count_zero_constant(self) -> None:
        repetitions = 0
        body_program = Loop(waveform=DummyWaveform(duration=1.0))
        body = DummyPulseTemplate(duration=2.0, program=body_program)

        # suppress warning about 0 repetitions on construction here, we are only interested in correct behavior during sequencing (i.e., do nothing)
        with warnings.catch_warnings(record=True):
            t = RepetitionPulseTemplate(body, repetitions)

        parameters = {}
        measurement_mapping = dict(moth='fire')
        channel_mapping = dict(asd='f')

        program = Loop()
        t._internal_create_program(parameters=parameters,
                                   measurement_mapping=measurement_mapping,
                                   channel_mapping=channel_mapping,
                                   to_single_waveform=set(),
                                   global_transformation=None,
                                   parent_loop=program)
        self.assertFalse(body.create_program_calls)
        self.assertFalse(program.children)
        self.assertEqual(1, program.repetition_count)
        self.assertEqual(None, program._measurements)

        # ensure same result as from Sequencer
        sequencer = Sequencer()
        sequencer.push(t, parameters=parameters, conditions={}, window_mapping=measurement_mapping,
                       channel_mapping=channel_mapping)
        block = sequencer.build()
        program_old = MultiChannelProgram(block, channels={'A'}).programs[frozenset({'A'})]
        self.assertEqual(program_old, program)

    def test_create_program_rep_count_zero_constant_with_measurement(self) -> None:
        repetitions = 0
        body_program = Loop(waveform=DummyWaveform(duration=1.0))
        body = DummyPulseTemplate(duration=2.0, program=body_program)

        # suppress warning about 0 repetitions on construction here, we are only interested in correct behavior during sequencing (i.e., do nothing)
        with warnings.catch_warnings(record=True):
            t = RepetitionPulseTemplate(body, repetitions, measurements=[('moth', 0, 'meas_end')])

        parameters = dict(meas_end=ConstantParameter(7.1))
        measurement_mapping = dict(moth='fire')
        channel_mapping = dict(asd='f')

        program = Loop()
        t._internal_create_program(parameters=parameters,
                                   measurement_mapping=measurement_mapping,
                                   channel_mapping=channel_mapping,
                                   to_single_waveform=set(),
                                   global_transformation=None,
                                   parent_loop=program)
        self.assertFalse(body.create_program_calls)
        self.assertFalse(program.children)
        self.assertEqual(1, program.repetition_count)
        self.assertEqual(None, program._measurements)

        # ensure same result as from Sequencer
        sequencer = Sequencer()
        sequencer.push(t, parameters=parameters, conditions={}, window_mapping=measurement_mapping,
                       channel_mapping=channel_mapping)
        block = sequencer.build()
        program_old = MultiChannelProgram(block, channels={'A'}).programs[frozenset({'A'})]
        self.assertEqual(program_old.repetition_count, program.repetition_count)
        self.assertEqual(program_old.waveform, program.waveform)
        self.assertEqual(program_old.children, program.children)
        # program_old will have measurements which program has not!

    def test_create_program_rep_count_zero_declaration(self) -> None:
        repetitions = "foo"
        body_program = Loop(waveform=DummyWaveform(duration=1.0))
        body = DummyPulseTemplate(duration=2.0, program=body_program)

        # suppress warning about 0 repetitions on construction here, we are only interested in correct behavior during sequencing (i.e., do nothing)
        with warnings.catch_warnings(record=True):
            t = RepetitionPulseTemplate(body, repetitions)

        parameters = dict(foo=ConstantParameter(0))
        measurement_mapping = dict(moth='fire')
        channel_mapping = dict(asd='f')

        program = Loop()
        t._internal_create_program(parameters=parameters,
                                   measurement_mapping=measurement_mapping,
                                   channel_mapping=channel_mapping,
                                   to_single_waveform=set(),
                                   global_transformation=None,
                                   parent_loop=program)
        self.assertFalse(body.create_program_calls)
        self.assertFalse(program.children)
        self.assertEqual(1, program.repetition_count)
        self.assertEqual(None, program._measurements)
        
        # ensure same result as from Sequencer
        sequencer = Sequencer()
        sequencer.push(t, parameters=parameters, conditions={}, window_mapping=measurement_mapping,
                       channel_mapping=channel_mapping)
        block = sequencer.build()
        program_old = MultiChannelProgram(block, channels={'A'}).programs[frozenset({'A'})]
        self.assertEqual(program_old, program)

    def test_create_program_rep_count_zero_declaration_with_measurement(self) -> None:
        repetitions = "foo"
        body_program = Loop(waveform=DummyWaveform(duration=1.0))
        body = DummyPulseTemplate(duration=2.0, program=body_program)

        # suppress warning about 0 repetitions on construction here, we are only interested in correct behavior during sequencing (i.e., do nothing)
        with warnings.catch_warnings(record=True):
            t = RepetitionPulseTemplate(body, repetitions, measurements=[('moth', 0, 'meas_end')])

        parameters = dict(foo=ConstantParameter(0), meas_end=ConstantParameter(7.1))
        measurement_mapping = dict(moth='fire')
        channel_mapping = dict(asd='f')

        program = Loop()
        t._internal_create_program(parameters=parameters,
                                   measurement_mapping=measurement_mapping,
                                   channel_mapping=channel_mapping,
                                   to_single_waveform=set(),
                                   global_transformation=None,
                                   parent_loop=program)
        self.assertFalse(body.create_program_calls)
        self.assertFalse(program.children)
        self.assertEqual(1, program.repetition_count)
        self.assertEqual(None, program._measurements)

        # ensure same result as from Sequencer
        sequencer = Sequencer()
        sequencer.push(t, parameters=parameters, conditions={}, window_mapping=measurement_mapping,
                       channel_mapping=channel_mapping)
        block = sequencer.build()
        program_old = MultiChannelProgram(block, channels={'A'}).programs[frozenset({'A'})]
        self.assertEqual(program_old.repetition_count, program.repetition_count)
        self.assertEqual(program_old.waveform, program.waveform)
        self.assertEqual(program_old.children, program.children)
        # program_old will have measurements which program has not!

    def test_create_program_rep_count_neg_declaration(self) -> None:
        repetitions = "foo"
        body_program = Loop(waveform=DummyWaveform(duration=1.0))
        body = DummyPulseTemplate(duration=2.0, program=body_program)

        # suppress warning about 0 repetitions on construction here, we are only interested in correct behavior during sequencing (i.e., do nothing)
        with warnings.catch_warnings(record=True):
            t = RepetitionPulseTemplate(body, repetitions)

        parameters = dict(foo=ConstantParameter(-1))
        measurement_mapping = dict(moth='fire')
        channel_mapping = dict(asd='f')

        program = Loop()
        t._internal_create_program(parameters=parameters,
                                   measurement_mapping=measurement_mapping,
                                   channel_mapping=channel_mapping,
                                   to_single_waveform=set(),
                                   global_transformation=None,
                                   parent_loop=program)
        self.assertFalse(body.create_program_calls)
        self.assertFalse(program.children)
        self.assertEqual(1, program.repetition_count)
        self.assertEqual(None, program._measurements)

        # ensure same result as from Sequencer
        sequencer = Sequencer()
        sequencer.push(t, parameters=parameters, conditions={}, window_mapping=measurement_mapping,
                       channel_mapping=channel_mapping)
        block = sequencer.build()
        program_old = MultiChannelProgram(block, channels={'A'}).programs[frozenset({'A'})]
        self.assertEqual(program_old, program)

    def test_create_program_rep_count_neg_declaration_with_measurements(self) -> None:
        repetitions = "foo"
        body_program = Loop(waveform=DummyWaveform(duration=1.0))
        body = DummyPulseTemplate(duration=2.0, program=body_program)

        # suppress warning about 0 repetitions on construction here, we are only interested in correct behavior during sequencing (i.e., do nothing)
        with warnings.catch_warnings(record=True):
            t = RepetitionPulseTemplate(body, repetitions, measurements=[('moth', 0, 'meas_end')])

        parameters = dict(foo=ConstantParameter(-1), meas_end=ConstantParameter(7.1))
        measurement_mapping = dict(moth='fire')
        channel_mapping = dict(asd='f')

        program = Loop()
        t._internal_create_program(parameters=parameters,
                                   measurement_mapping=measurement_mapping,
                                   channel_mapping=channel_mapping,
                                   to_single_waveform=set(),
                                   global_transformation=None,
                                   parent_loop=program)
        self.assertFalse(body.create_program_calls)
        self.assertFalse(program.children)
        self.assertEqual(1, program.repetition_count)
        self.assertEqual(None, program._measurements)

        # ensure same result as from Sequencer
        sequencer = Sequencer()
        sequencer.push(t, parameters=parameters, conditions={}, window_mapping=measurement_mapping,
                       channel_mapping=channel_mapping)
        block = sequencer.build()
        program_old = MultiChannelProgram(block, channels={'A'}).programs[frozenset({'A'})]
        self.assertEqual(program_old.repetition_count, program.repetition_count)
        self.assertEqual(program_old.waveform, program.waveform)
        self.assertEqual(program_old.children, program.children)
        # program_old will have measurements which program has not!

    def test_create_program_none_subprogram(self) -> None:
        repetitions = "foo"
        body = DummyPulseTemplate(duration=0.0, waveform=None)
        t = RepetitionPulseTemplate(body, repetitions, parameter_constraints=['foo<9'])
        parameters = dict(foo=ConstantParameter(3))
        measurement_mapping = dict(moth='fire')
        channel_mapping = dict(asd='f')
        program = Loop()
        t._internal_create_program(parameters=parameters,
                                   measurement_mapping=measurement_mapping,
                                   channel_mapping=channel_mapping,
                                   to_single_waveform=set(),
                                   global_transformation=None,
                                   parent_loop=program)
        self.assertFalse(program.children)
        self.assertEqual(1, program.repetition_count)
        self.assertEqual(None, program._measurements)

        # ensure same result as from Sequencer
        sequencer = Sequencer()
        sequencer.push(t, parameters=parameters, conditions={}, window_mapping=measurement_mapping,
                       channel_mapping=channel_mapping)
        block = sequencer.build()
        program_old = MultiChannelProgram(block, channels={'A'}).programs[frozenset({'A'})]
        self.assertEqual(program_old.waveform, program.waveform)
        self.assertEqual(program_old.children, program.children)
        self.assertEqual(program_old._measurements, program._measurements)
        # Sequencer does set a repetition count if no inner program is present; create_program does not

    def test_create_program_none_subprogram_with_measurement(self) -> None:
        repetitions = "foo"
        body = DummyPulseTemplate(duration=2.0, waveform=None, measurements=[('b', 2, 3)])
        t = RepetitionPulseTemplate(body, repetitions, parameter_constraints=['foo<9'], measurements=[('moth', 0, 'meas_end')])
        parameters = dict(foo=ConstantParameter(3), meas_end=ConstantParameter(7.1))
        measurement_mapping = dict(moth='fire', b='b')
        channel_mapping = dict(asd='f')
        program = Loop()
        t._internal_create_program(parameters=parameters,
                                   measurement_mapping=measurement_mapping,
                                   channel_mapping=channel_mapping,
                                   to_single_waveform=set(),
                                   global_transformation=None,
                                   parent_loop=program)
        self.assertFalse(program.children)
        self.assertEqual(1, program.repetition_count)
        self.assertEqual(None, program._measurements)

        # ensure same result as from Sequencer
        sequencer = Sequencer()
        sequencer.push(t, parameters=parameters, conditions={}, window_mapping=measurement_mapping,
                       channel_mapping=channel_mapping)
        block = sequencer.build()
        program_old = MultiChannelProgram(block, channels={'A'}).programs[frozenset({'A'})]
        self.assertEqual(program_old.waveform, program.waveform)
        self.assertEqual(program_old.children, program.children)
        # program_old will have measurements which program has not!
        # Sequencer does set a repetition count if no inner program is present; create_program does not


class RepetitionPulseTemplateOldSequencingTests(unittest.TestCase):

    def setUp(self) -> None:
        self.body = DummyPulseTemplate()
        self.repetitions = 'foo'
        self.template = RepetitionPulseTemplate(self.body, self.repetitions, parameter_constraints=['foo<9'])
        self.sequencer = DummySequencer()
        self.block = DummyInstructionBlock()

    def test_build_sequence_constant(self) -> None:
        repetitions = 3
        t = RepetitionPulseTemplate(self.body, repetitions)
        parameters = {}
        measurement_mapping = {'my': 'thy'}
        conditions = dict(foo=DummyCondition(requires_stop=True))
        channel_mapping = {}
        t.build_sequence(self.sequencer, parameters, conditions, measurement_mapping, channel_mapping, self.block)

        self.assertTrue(self.block.embedded_blocks)
        body_block = self.block.embedded_blocks[0]
        self.assertEqual({body_block}, set(self.sequencer.sequencing_stacks.keys()))
        self.assertEqual([(self.body, parameters, conditions, measurement_mapping, channel_mapping)], self.sequencer.sequencing_stacks[body_block])
        self.assertEqual([REPJInstruction(repetitions, InstructionPointer(body_block, 0))], self.block.instructions)

    def test_build_sequence_declaration_success(self) -> None:
        parameters = dict(foo=ConstantParameter(3))
        conditions = dict(foo=DummyCondition(requires_stop=True))
        measurement_mapping = dict(moth='fire')
        channel_mapping = dict(asd='f')
        self.template.build_sequence(self.sequencer, parameters, conditions, measurement_mapping, channel_mapping, self.block)

        self.assertTrue(self.block.embedded_blocks)
        body_block = self.block.embedded_blocks[0]
        self.assertEqual({body_block}, set(self.sequencer.sequencing_stacks.keys()))
        self.assertEqual([(self.body, parameters, conditions, measurement_mapping, channel_mapping)],
                         self.sequencer.sequencing_stacks[body_block])
        self.assertEqual([REPJInstruction(3, InstructionPointer(body_block, 0))], self.block.instructions)

    def test_parameter_not_provided(self):
        parameters = dict(foo=ConstantParameter(4))
        conditions = dict(foo=DummyCondition(requires_stop=True))
        measurement_mapping = dict(moth='fire')
        channel_mapping = dict(asd='f')

        template = RepetitionPulseTemplate(self.body, 'foo*bar', parameter_constraints=['foo<9'])

        with self.assertRaises(ParameterNotProvidedException):
            template.build_sequence(self.sequencer, parameters, conditions, measurement_mapping, channel_mapping,
                                     self.block)

    def test_build_sequence_declaration_exceeds_bounds(self) -> None:
        parameters = dict(foo=ConstantParameter(9))
        conditions = dict(foo=DummyCondition(requires_stop=True))
        with self.assertRaises(ParameterConstraintViolation):
            self.template.build_sequence(self.sequencer, parameters, conditions, {}, {}, self.block)
        self.assertFalse(self.sequencer.sequencing_stacks)

    def test_build_sequence_declaration_parameter_missing(self) -> None:
        parameters = {}
        conditions = dict(foo=DummyCondition(requires_stop=True))
        with self.assertRaises(ParameterNotProvidedException):
            self.template.build_sequence(self.sequencer, parameters, conditions, {}, {}, self.block)
        self.assertFalse(self.sequencer.sequencing_stacks)

    def test_build_sequence_declaration_parameter_value_not_whole(self) -> None:
        parameters = dict(foo=ConstantParameter(3.3))
        conditions = dict(foo=DummyCondition(requires_stop=True))
        with self.assertRaises(ParameterNotIntegerException):
            self.template.build_sequence(self.sequencer, parameters, conditions, {}, {}, self.block)
        self.assertFalse(self.sequencer.sequencing_stacks)

    def test_rep_count_zero_constant(self) -> None:
        repetitions = 0
        parameters = {}
        measurement_mapping = {}
        conditions = {}
        channel_mapping = {}

        # suppress warning about 0 repetitions on construction here, we are only interested in correct behavior during sequencing (i.e., do nothing)
        with warnings.catch_warnings(record=True):
            t = RepetitionPulseTemplate(self.body, repetitions)
            t.build_sequence(self.sequencer, parameters, conditions, measurement_mapping, channel_mapping, self.block)

            self.assertFalse(self.block.embedded_blocks) # no new blocks created
            self.assertFalse(self.block.instructions) # no instructions added to block

    def test_rep_count_zero_declaration(self) -> None:
        t = self.template
        parameters = dict(foo=ConstantParameter(0))
        measurement_mapping = {}
        conditions = {}
        channel_mapping = {}
        t.build_sequence(self.sequencer, parameters, conditions, measurement_mapping, channel_mapping, self.block)

        self.assertFalse(self.block.embedded_blocks) # no new blocks created
        self.assertFalse(self.block.instructions) # no instructions added to block

    def test_rep_count_neg_declaration(self) -> None:
        t = self.template
        parameters = dict(foo=ConstantParameter(-1))
        measurement_mapping = {}
        conditions = {}
        channel_mapping = {}
        t.build_sequence(self.sequencer, parameters, conditions, measurement_mapping, channel_mapping, self.block)

        self.assertFalse(self.block.embedded_blocks)  # no new blocks created
        self.assertFalse(self.block.instructions)  # no instructions added to block

    def test_requires_stop_constant(self) -> None:
        body = DummyPulseTemplate(requires_stop=False)
        t = RepetitionPulseTemplate(body, 2)
        self.assertFalse(t.requires_stop({}, {}))
        body.requires_stop_ = True
        self.assertFalse(t.requires_stop({}, {}))

    def test_requires_stop_declaration(self) -> None:
        body = DummyPulseTemplate(requires_stop=False)
        t = RepetitionPulseTemplate(body, 'foo')

        parameter = DummyParameter()
        parameters = dict(foo=parameter)
        condition = DummyCondition()
        conditions = dict(foo=condition)

        for body_requires_stop in [True, False]:
            for condition_requires_stop in [True, False]:
                for parameter_requires_stop in [True, False]:
                    body.requires_stop_ = body_requires_stop
                    condition.requires_stop_ = condition_requires_stop
                    parameter.requires_stop_ = parameter_requires_stop
                    self.assertEqual(parameter_requires_stop, t.requires_stop(parameters, conditions))


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