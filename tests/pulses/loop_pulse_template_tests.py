import unittest
from unittest import mock

from qupulse.expressions import Expression, ExpressionScalar
from qupulse.pulses.loop_pulse_template import ForLoopPulseTemplate, WhileLoopPulseTemplate,\
    ConditionMissingException, ParametrizedRange, LoopIndexNotUsedException, LoopPulseTemplate
from qupulse.pulses.parameters import ConstantParameter, InvalidParameterNameException, ParameterConstraintViolation,\
    ParameterNotProvidedException, ParameterConstraint
from qupulse._program.instructions import MEASInstruction
from qupulse._program._loop import Loop

from qupulse._program._loop import MultiChannelProgram
from qupulse.pulses.sequencing import Sequencer

from tests.pulses.sequencing_dummies import DummyCondition, DummyPulseTemplate, DummySequencer, DummyInstructionBlock,\
    DummyParameter, MeasurementWindowTestCase, DummyWaveform
from tests.serialization_dummies import DummySerializer
from tests.serialization_tests import SerializableTests
from tests._program.transformation_tests import TransformationStub


class DummyLoopPulseTemplate(LoopPulseTemplate):
    pass
DummyLoopPulseTemplate.__abstractmethods__ = set()
DummyLoopPulseTemplate.__init__ = lambda self, body: LoopPulseTemplate.__init__(self, body, identifier=None)


class LoopPulseTemplateTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_body(self):
        body = DummyPulseTemplate()
        tpl = DummyLoopPulseTemplate(body)
        self.assertIs(tpl.body, body)

    def test_defined_channels(self):
        body = DummyPulseTemplate(defined_channels={'A'})
        tpl = DummyLoopPulseTemplate(body)
        self.assertEqual(tpl.defined_channels, body.defined_channels)

    def test_measurement_names(self):
        body = DummyPulseTemplate(measurement_names={'A'})
        tpl = DummyLoopPulseTemplate(body)
        self.assertIs(tpl.measurement_names, body.measurement_names)


class ParametrizedRangeTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_init(self):
        self.assertEqual(ParametrizedRange(7).to_tuple(),
                         (0, 7, 1))
        self.assertEqual(ParametrizedRange(4, 7).to_tuple(),
                         (4, 7, 1))
        self.assertEqual(ParametrizedRange(4, 'h', 5).to_tuple(),
                         (4, 'h', 5))

        self.assertEqual(ParametrizedRange(start=7, stop=1, step=-1).to_tuple(),
                         (7, 1, -1))

        with self.assertRaises(TypeError):
            ParametrizedRange()
        with self.assertRaises(TypeError):
            ParametrizedRange(1, 2, 3, 4)

        with self.assertRaises(TypeError):
            ParametrizedRange(1, 2, stop=6)

    def test_to_range(self):
        pr = ParametrizedRange(4, 'l*k', 'k')

        self.assertEqual(pr.to_range({'l': 5, 'k': 2}),
                         range(4, 10, 2))

    def test_parameter_names(self):
        self.assertEqual(ParametrizedRange(5).parameter_names, set())
        self.assertEqual(ParametrizedRange('g').parameter_names, {'g'})
        self.assertEqual(ParametrizedRange('g*h', 'h', 'l/m').parameter_names, {'g', 'h', 'l', 'm'})


class ForLoopPulseTemplateTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_init(self):
        dt = DummyPulseTemplate(parameter_names={'i', 'k'})
        self.assertEqual(ForLoopPulseTemplate(body=dt, loop_index='i', loop_range=5).loop_range.to_tuple(),
                         (0, 5, 1))
        self.assertEqual(ForLoopPulseTemplate(body=dt, loop_index='i', loop_range='s').loop_range.to_tuple(),
                         (0, 's', 1))
        self.assertEqual(ForLoopPulseTemplate(body=dt, loop_index='i', loop_range=(2, 5)).loop_range.to_tuple(),
                         (2, 5, 1))
        self.assertEqual(ForLoopPulseTemplate(body=dt, loop_index='i', loop_range=range(1, 2, 5)).loop_range.to_tuple(),
                         (1, 2, 5))
        self.assertEqual(ForLoopPulseTemplate(body=dt, loop_index='i',
                                              loop_range=ParametrizedRange('a', 'b', 'c')).loop_range.to_tuple(),
                         ('a', 'b', 'c'))

        with self.assertRaises(InvalidParameterNameException):
            ForLoopPulseTemplate(body=dt, loop_index='1i', loop_range=6)

        with self.assertRaises(ValueError):
            ForLoopPulseTemplate(body=dt, loop_index='i', loop_range=slice(None))

        with self.assertRaises(LoopIndexNotUsedException):
            ForLoopPulseTemplate(body=DummyPulseTemplate(), loop_index='i', loop_range=1)

    def test_body_parameter_generator(self):
        dt = DummyPulseTemplate(parameter_names={'i', 'k'})
        flt = ForLoopPulseTemplate(body=dt, loop_index='i', loop_range=('a', 'b', 'c'))

        expected_range = range(2, 17, 3)

        outer_params = dict(k=ConstantParameter(5),
                            a=ConstantParameter(expected_range.start),
                            b=ConstantParameter(expected_range.stop),
                            c=ConstantParameter(expected_range.step))
        forward_parameter_dicts = list(flt._body_parameter_generator(outer_params, forward=True))
        backward_parameter_dicts = list(flt._body_parameter_generator(outer_params, forward=False))

        self.assertEqual(forward_parameter_dicts, list(reversed(backward_parameter_dicts)))
        for local_params, i in zip(forward_parameter_dicts, expected_range):
            expected_local_params = dict(k=ConstantParameter(5), i=ConstantParameter(i))
            self.assertEqual(expected_local_params, local_params)

    def test_loop_index(self):
        loop_index = 'i'
        dt = DummyPulseTemplate(parameter_names={'i', 'k'})
        flt = ForLoopPulseTemplate(body=dt, loop_index=loop_index, loop_range=('a', 'b', 'c'))
        self.assertIs(loop_index, flt.loop_index)

    def test_duration(self):
        dt = DummyPulseTemplate(parameter_names={'idx', 'd'}, duration=Expression('d+idx*2'))

        flt = ForLoopPulseTemplate(body=dt, loop_index='idx', loop_range='n')
        self.assertEqual(flt.duration.evaluate_numeric(n=4, d=100), 4 * 100 + 2 * (1 + 2 + 3))

        flt = ForLoopPulseTemplate(body=dt, loop_index='idx', loop_range=(3, 'n', 2))
        self.assertEqual(flt.duration.evaluate_numeric(n=9, d=100), 3*100 + 2*(3 + 5 + 7))
        self.assertEqual(flt.duration.evaluate_numeric(n=8, d=100), 3 * 100 + 2 * (3 + 5 + 7))

        flt = ForLoopPulseTemplate(body=dt, loop_index='idx', loop_range=('m', 'n', -2))
        self.assertEqual(flt.duration.evaluate_numeric(n=9, d=100, m=14),
                         3 * 100 + 2 * (14 + 12 + 10))

        flt = ForLoopPulseTemplate(body=dt, loop_index='idx', loop_range=('m', 'n', -2))
        self.assertEqual(flt.duration.evaluate_numeric(n=9, d=100, m=14),
                         3 * 100 + 2*(14 + 12 + 10))

    def test_parameter_names(self):
        dt = DummyPulseTemplate(parameter_names={'i', 'k'})
        flt = ForLoopPulseTemplate(body=dt, loop_index='i', loop_range=('a', 'b', 'c'),
                                   parameter_constraints={'c > hugo'}, measurements=[('meas', 'd', 1)])
        self.assertEqual({'k', 'a', 'b', 'c', 'd', 'hugo'}, flt.parameter_names)

    def test_parameter_names_param_only_in_constraint(self) -> None:
        flt = ForLoopPulseTemplate(body=DummyPulseTemplate(parameter_names={'k', 'i'}), loop_index='i',
                                   loop_range=('a', 'b', 'c',), parameter_constraints=['k<=f'])
        self.assertEqual(flt.parameter_names, {'k', 'a', 'b', 'c', 'f'})

    def test_integral(self) -> None:
        dummy = DummyPulseTemplate(defined_channels={'A', 'B'},
                                   parameter_names={'t1', 'i'},
                                   integrals={'A': ExpressionScalar('t1-i*3.1'), 'B': ExpressionScalar('i')})

        pulse = ForLoopPulseTemplate(dummy, 'i', (1, 8, 2))

        expected = {'A': ExpressionScalar('Sum(t1-3.1*(1+2*i), (i, 0, 3))'),
                    'B': ExpressionScalar('Sum((1+2*i), (i, 0, 3))') }
        self.assertEqual(expected, pulse.integral)


class ForLoopTemplateSequencingTests(MeasurementWindowTestCase):

    def test_create_program_constraint_on_loop_var_exception(self):
        """This test is to assure the status-quo behavior of ForLoopPT handling parameter constraints affecting the loop index
        variable. Please see https://github.com/qutech/qupulse/issues/232 ."""

        with self.assertWarnsRegex(UserWarning, "constraint on a variable shadowing the loop index",
                                   msg="ForLoopPT did not issue a warning when constraining the loop index"):
            flt = ForLoopPulseTemplate(body=DummyPulseTemplate(parameter_names={'k', 'i'}), loop_index='i',
                                       loop_range=('a', 'b', 'c',), parameter_constraints=['k<=f', 'k>i'])

        # loop index showing up in parameter_names because it appears in consraints
        self.assertEqual(flt.parameter_names, {'f', 'k', 'a', 'b', 'c', 'i'})

        parameters = {'k': ConstantParameter(1), 'a': ConstantParameter(0), 'b': ConstantParameter(2),
                      'c': ConstantParameter(1), 'f': ConstantParameter(2)}

        # loop index not accessible in current build_sequence -> Exception
        children = [Loop(waveform=DummyWaveform(duration=2.0))]
        program = Loop(children=children)

        with self.assertRaises(ParameterNotProvidedException):
            flt._internal_create_program(parameters=parameters,
                                         measurement_mapping=dict(),
                                         channel_mapping=dict(),
                                         parent_loop=program,
                                         to_single_waveform=set(),
                                         global_transformation=None)
        self.assertEqual(children, program.children)
        self.assertEqual(1, program.repetition_count)
        self.assertIsNone(program._measurements)
        self.assert_measurement_windows_equal({}, program.get_measurement_windows())

    def test_create_program_invalid_params(self) -> None:
        dt = DummyPulseTemplate(parameter_names={'i'}, waveform=DummyWaveform(duration=4.0), duration=4, measurements=[('b', 2, 1)])
        flt = ForLoopPulseTemplate(body=dt, loop_index='i', loop_range=('a', 'b', 'c'),
                                   measurements=[('A', 0, 1)], parameter_constraints=['c > 1'])

        invalid_parameters = {'a': ConstantParameter(1), 'b': ConstantParameter(4), 'c': ConstantParameter(1)}
        measurement_mapping = dict(A='B')
        channel_mapping = dict(C='D')

        children = [Loop(waveform=DummyWaveform(duration=2.0))]
        program = Loop(children=children)
        with self.assertRaises(ParameterConstraintViolation):
            flt._internal_create_program(parameters=invalid_parameters,
                                         measurement_mapping=measurement_mapping,
                                         channel_mapping=channel_mapping,
                                         parent_loop=program,
                                         to_single_waveform=set(),
                                         global_transformation=None)

        self.assertEqual(children, program.children)
        self.assertEqual(1, program.repetition_count)
        self.assertIsNone(program._measurements)
        self.assert_measurement_windows_equal({}, program.get_measurement_windows())

    def test_create_program_invalid_measurement_mapping(self) -> None:
        dt = DummyPulseTemplate(parameter_names={'i'}, waveform=DummyWaveform(duration=4.0), duration=4,
                                measurements=[('b', 2, 1)])
        flt = ForLoopPulseTemplate(body=dt, loop_index='i', loop_range=('a', 'b', 'c'),
                                   measurements=[('A', 0, 1)], parameter_constraints=['c > 1'])

        invalid_parameters = {'a': ConstantParameter(1), 'b': ConstantParameter(4), 'c': ConstantParameter(2)}
        measurement_mapping = dict()
        channel_mapping = dict(C='D')

        children = [Loop(waveform=DummyWaveform(duration=2.0))]
        program = Loop(children=children)
        with self.assertRaises(KeyError):
            flt._internal_create_program(parameters=invalid_parameters,
                                         measurement_mapping=measurement_mapping,
                                         channel_mapping=channel_mapping,
                                         parent_loop=program,
                                         to_single_waveform=set(),
                                         global_transformation=None)

        self.assertEqual(children, program.children)
        self.assertEqual(1, program.repetition_count)
        self.assertIsNone(program._measurements)
        self.assert_measurement_windows_equal({}, program.get_measurement_windows())

        # test for broken mapping on child level. no guarantee that parent_loop is not changed, only check for exception
        measurement_mapping = dict(A='B')
        with self.assertRaises(KeyError):
            flt._internal_create_program(parameters=invalid_parameters,
                                         measurement_mapping=measurement_mapping,
                                         channel_mapping=channel_mapping,
                                         parent_loop=program,
                                         to_single_waveform=set(),
                                         global_transformation=None)

    def test_create_program_missing_params(self) -> None:
        dt = DummyPulseTemplate(parameter_names={'i'}, waveform=DummyWaveform(duration=4.0), duration='t', measurements=[('b', 2, 1)])
        flt = ForLoopPulseTemplate(body=dt, loop_index='i', loop_range=('a', 'b', 'c'),
                                   measurements=[('A', 'alph', 1)], parameter_constraints=['c > 1'])

        parameters = {'a': ConstantParameter(1), 'b': ConstantParameter(4)}
        measurement_mapping = dict(A='B')
        channel_mapping = dict(C='D')

        children = [Loop(waveform=DummyWaveform(duration=2.0))]
        program = Loop(children=children)

        # test parameter in constraints
        with self.assertRaises(ParameterNotProvidedException):
            flt._internal_create_program(parameters=parameters,
                                         measurement_mapping=measurement_mapping,
                                         channel_mapping=channel_mapping,
                                         parent_loop=program,
                                         to_single_waveform=set(),
                                         global_transformation=None)

        # test parameter in measurement mappings
        parameters = {'a': ConstantParameter(1), 'b': ConstantParameter(4), 'c': ConstantParameter(2)}
        with self.assertRaises(ParameterNotProvidedException):
            flt._internal_create_program(parameters=parameters,
                                         measurement_mapping=measurement_mapping,
                                         channel_mapping=channel_mapping,
                                         parent_loop=program,
                                         to_single_waveform=set(),
                                         global_transformation=None)

        # test parameter in duration
        parameters = {'a': ConstantParameter(1), 'b': ConstantParameter(4), 'c': ConstantParameter(2), 'alph': ConstantParameter(0)}
        with self.assertRaises(ParameterNotProvidedException):
            flt._internal_create_program(parameters=parameters,
                                         measurement_mapping=measurement_mapping,
                                         channel_mapping=channel_mapping,
                                         parent_loop=program,
                                         to_single_waveform=set(),
                                         global_transformation=None)

        self.assertEqual(children, program.children)
        self.assertEqual(1, program.repetition_count)
        self.assertIsNone(program._measurements)
        self.assert_measurement_windows_equal({}, program.get_measurement_windows())

    def test_create_program_body_none(self) -> None:
        dt = DummyPulseTemplate(parameter_names={'i'}, waveform=None, duration=0,
                                measurements=[('b', 2, 1)])
        flt = ForLoopPulseTemplate(body=dt, loop_index='i', loop_range=('a', 'b', 'c'),
                                   measurements=[('A', 0, 1)], parameter_constraints=['c > 1'])

        parameters = {'a': ConstantParameter(1), 'b': ConstantParameter(4), 'c': ConstantParameter(2)}
        measurement_mapping = dict(A='B', b='b')
        channel_mapping = dict(C='D')

        program = Loop()
        flt._internal_create_program(parameters=parameters,
                                     measurement_mapping=measurement_mapping,
                                     channel_mapping=channel_mapping,
                                     parent_loop=program,
                                     to_single_waveform=set(),
                                     global_transformation=None)

        self.assertEqual(0, len(program.children))
        self.assertEqual(1, program.repetition_count)
        self.assertEqual([], program.children)

        # ensure same result as from Sequencer
        sequencer = Sequencer()
        sequencer.push(flt, parameters=parameters, conditions={}, window_mapping=measurement_mapping,
                       channel_mapping=channel_mapping)
        block = sequencer.build()
        program_old = MultiChannelProgram(block, channels={'A'}).programs[frozenset({'A'})]
        self.assertEqual(program_old.repetition_count, program.repetition_count)
        self.assertEqual(program_old.children, program.children)
        self.assertEqual(program_old.waveform, program.waveform)
        # program_old defines measurements while program does not

    def test_create_program(self) -> None:
        dt = DummyPulseTemplate(parameter_names={'i'},
                                waveform=DummyWaveform(duration=4.0, defined_channels={'A'}),
                                duration=4,
                                measurements=[('b', .2, .3)])
        flt = ForLoopPulseTemplate(body=dt, loop_index='i', loop_range=('a', 'b', 'c'),
                                   measurements=[('A', 'meas_param', 1)], parameter_constraints=['c > 1'])

        parameters = {'a': ConstantParameter(1), 'b': ConstantParameter(4), 'c': ConstantParameter(2),
                      'meas_param': ConstantParameter(.1)}
        measurement_mapping = dict(A='B', b='b')
        channel_mapping = dict(C='D')

        to_single_waveform = {'tom', 'jerry'}
        global_transformation = TransformationStub()

        expected_meas_params = {'meas_param': .1}

        program = Loop()

        # inner _create_program does nothing
        expected_program = Loop(measurements=[('B', .1, 1)])

        expected_create_program_kwargs = dict(measurement_mapping=measurement_mapping,
                                              channel_mapping=channel_mapping,
                                              global_transformation=global_transformation,
                                              to_single_waveform=to_single_waveform,
                                              parent_loop=program)
        expected_create_program_calls = [mock.call(**expected_create_program_kwargs,
                                                   parameters=dict(i=ConstantParameter(i)))
                                         for i in (1, 3)]

        with mock.patch.object(flt, 'validate_parameter_constraints') as validate_parameter_constraints:
            with mock.patch.object(dt, '_create_program') as body_create_program:
                with mock.patch.object(flt, 'get_measurement_windows',
                                       wraps=flt.get_measurement_windows) as get_measurement_windows:
                    flt._internal_create_program(parameters=parameters,
                                                 measurement_mapping=measurement_mapping,
                                                 channel_mapping=channel_mapping,
                                                 parent_loop=program,
                                                 to_single_waveform=to_single_waveform,
                                                 global_transformation=global_transformation)

                    validate_parameter_constraints.assert_called_once_with(parameters=parameters)
                    get_measurement_windows.assert_called_once_with(expected_meas_params, measurement_mapping)
                    self.assertEqual(body_create_program.call_args_list, expected_create_program_calls)

        self.assertEqual(expected_program, program)

    def test_create_program_append(self) -> None:
        dt = DummyPulseTemplate(parameter_names={'i'}, waveform=DummyWaveform(duration=4.0), duration=4,
                                measurements=[('b', 2, 1)])
        flt = ForLoopPulseTemplate(body=dt, loop_index='i', loop_range=('a', 'b', 'c'),
                                   measurements=[('A', 0, 1)], parameter_constraints=['c > 1'])

        parameters = {'a': ConstantParameter(1), 'b': ConstantParameter(4), 'c': ConstantParameter(2)}
        measurement_mapping = dict(A='B', b='b')
        channel_mapping = dict(C='D')

        children = [Loop(waveform=DummyWaveform(duration=2.0))]
        program = Loop(children=children)
        flt._internal_create_program(parameters=parameters,
                                     measurement_mapping=measurement_mapping,
                                     channel_mapping=channel_mapping,
                                     parent_loop=program,
                                     to_single_waveform=set(),
                                     global_transformation=None)

        self.assertEqual(3, len(program.children))
        self.assertIs(children[0], program.children[0])
        self.assertEqual(dt.waveform, program.children[1].waveform)
        self.assertEqual(dt.waveform, program.children[2].waveform)
        self.assertEqual(1, program.children[1].repetition_count)
        self.assertEqual(1, program.children[2].repetition_count)
        self.assertEqual(1, program.repetition_count)
        self.assert_measurement_windows_equal({'b': ([4, 8], [1, 1]), 'B': ([2], [1])}, program.get_measurement_windows())

        # not ensure same result as from Sequencer here - we're testing appending to an already existing parent loop
        # which is a use case that does not immediately arise from using Sequencer


class ForLoopTemplateOldSequencingTests(unittest.TestCase):

    def test_build_sequence_constraint_on_loop_var_exception(self):
        """This test is to assure the status-quo behavior of ForLoopPT handling parameter constraints affecting the loop index
        variable. Please see https://github.com/qutech/qupulse/issues/232 ."""

        with self.assertWarnsRegex(UserWarning, "constraint on a variable shadowing the loop index",
                                   msg="ForLoopPT did not issue a warning when constraining the loop index"):
            flt = ForLoopPulseTemplate(body=DummyPulseTemplate(parameter_names={'k', 'i'}), loop_index='i',
                                       loop_range=('a', 'b', 'c',), parameter_constraints=['k<=f', 'k>i'])

        # loop index showing up in parameter_names because it appears in consraints
        self.assertEqual(flt.parameter_names, {'f', 'k', 'a', 'b', 'c', 'i'})

        parameters = {'k': ConstantParameter(1), 'a': ConstantParameter(0), 'b': ConstantParameter(2),
                      'c': ConstantParameter(1), 'f': ConstantParameter(2)}
        sequencer = DummySequencer()
        block = DummyInstructionBlock()

        # loop index not accessible in current build_sequence -> Exception
        self.assertRaises(ParameterNotProvidedException, flt.build_sequence, sequencer, parameters, dict(), dict(), dict(), block)

    def test_build_sequence(self):
        dt = DummyPulseTemplate(parameter_names={'i'})
        flt = ForLoopPulseTemplate(body=dt, loop_index='i', loop_range=('a', 'b', 'c'),
                                   measurements=[('A', 0, 1)], parameter_constraints=['c > 1'])

        sequencer = DummySequencer()
        block = DummyInstructionBlock()
        invalid_parameters = {'a': ConstantParameter(1), 'b': ConstantParameter(4), 'c': ConstantParameter(1)}
        parameters = {'a': ConstantParameter(1), 'b': ConstantParameter(4), 'c': ConstantParameter(2)}
        measurement_mapping = dict(A='B')
        channel_mapping = dict(C='D')

        with self.assertRaises(ParameterConstraintViolation):
            flt.build_sequence(sequencer, invalid_parameters, dict(), measurement_mapping, channel_mapping, block)

        self.assertEqual(block.instructions, [])
        self.assertNotIn(block, sequencer.sequencing_stacks)

        flt.build_sequence(sequencer, parameters, dict(), measurement_mapping, channel_mapping, block)

        self.assertEqual(block.instructions, [MEASInstruction(measurements=[('B', 0, 1)])])

        expected_stack = [(dt, {'i': ConstantParameter(3)}, dict(), measurement_mapping, channel_mapping),
                          (dt, {'i': ConstantParameter(1)}, dict(), measurement_mapping, channel_mapping)]

        self.assertEqual(sequencer.sequencing_stacks[block], expected_stack)

    def test_requires_stop(self):
        parameters = dict(A=DummyParameter(requires_stop=False), B=DummyParameter(requires_stop=False))

        dt = DummyPulseTemplate(parameter_names={'i'})
        flt = ForLoopPulseTemplate(body=dt, loop_index='i', loop_range=('A', 'B'))

        self.assertFalse(flt.requires_stop(parameters, dict()))

        parameters['A'] = DummyParameter(requires_stop=True)
        self.assertTrue(flt.requires_stop(parameters, dict()))


class ForLoopPulseTemplateSerializationTests(SerializableTests, unittest.TestCase):

    @property
    def class_to_test(self):
        return ForLoopPulseTemplate

    def make_kwargs(self):
        return {
            'body': DummyPulseTemplate(parameter_names={'i'}),
            'loop_index': 'i',
            'loop_range': ('A', 'B', 1),
            'parameter_constraints': [str(ParameterConstraint('foo < 3'))],
            'measurements': [('a', 0, 1), ('b', 1, 1)]
        }

    def assert_equal_instance_except_id(self, lhs: ForLoopPulseTemplate, rhs: ForLoopPulseTemplate):
        self.assertIsInstance(lhs, ForLoopPulseTemplate)
        self.assertIsInstance(rhs, ForLoopPulseTemplate)
        self.assertEqual(lhs.body, rhs.body)
        self.assertEqual(lhs.loop_index, rhs.loop_index)
        self.assertEqual(lhs.loop_range.to_tuple(), rhs.loop_range.to_tuple())
        self.assertEqual(lhs.parameter_constraints, rhs.parameter_constraints)
        self.assertEqual(lhs.measurement_declarations, rhs.measurement_declarations)


class ForLoopPulseTemplateOldSerializationTests(unittest.TestCase):

    def test_get_serialization_data_minimal_old(self) -> None:
        # test for deprecated version during transition period, remove after final switch
        with self.assertWarnsRegex(DeprecationWarning, "deprecated",
                                   msg="ForLoopPT does not issue warning for old serialization routines."):
            dt = DummyPulseTemplate(parameter_names={'i'})
            flt = ForLoopPulseTemplate(body=dt, loop_index='i', loop_range=('A', 'B'))

            def check_dt(to_dictify) -> str:
                self.assertIs(to_dictify, dt)
                return 'dt'

            serializer = DummySerializer(serialize_callback=check_dt)

            data = flt.get_serialization_data(serializer)
            expected_data = dict(body='dt',
                                 loop_range=('A', 'B', 1),
                                 loop_index='i')
            self.assertEqual(data, expected_data)

    def test_get_serialization_data_all_features_old(self) -> None:
        # test for deprecated version during transition period, remove after final switch
        with self.assertWarnsRegex(DeprecationWarning, "deprecated",
                                   msg="ForLoopPT does not issue warning for old serialization routines."):
            measurements = [('a', 0, 1), ('b', 1, 1)]
            parameter_constraints = ['foo < 3']

            dt = DummyPulseTemplate(parameter_names={'i'})
            flt = ForLoopPulseTemplate(body=dt, loop_index='i', loop_range=('A', 'B'),
                                       measurements=measurements, parameter_constraints=parameter_constraints)

            def check_dt(to_dictify) -> str:
                self.assertIs(to_dictify, dt)
                return 'dt'

            serializer = DummySerializer(serialize_callback=check_dt)

            data = flt.get_serialization_data(serializer)
            expected_data = dict(body='dt',
                                 loop_range=('A', 'B', 1),
                                 loop_index='i',
                                 measurements=measurements,
                                 parameter_constraints=parameter_constraints)
            self.assertEqual(data, expected_data)

    def test_deserialize_minimal_old(self) -> None:
        # test for deprecated version during transition period, remove after final switch
        with self.assertWarnsRegex(DeprecationWarning, "deprecated",
                                   msg="ForLoopPT does not issue warning for old serialization routines."):
            body_str = 'dt'
            dt = DummyPulseTemplate(parameter_names={'i'})

            def make_dt(ident: str):
                self.assertEqual(body_str, ident)
                return ident

            data = dict(body=body_str,
                        loop_range=('A', 'B', 1),
                        loop_index='i',
                        identifier='meh')

            serializer = DummySerializer(deserialize_callback=make_dt)
            serializer.subelements['dt'] = dt

            flt = ForLoopPulseTemplate.deserialize(serializer, **data)
            self.assertEqual(flt.identifier, 'meh')
            self.assertEqual(flt.body, dt)
            self.assertEqual(flt.loop_index, 'i')
            self.assertEqual(flt.loop_range.to_tuple(), ('A', 'B', 1))

    def test_deserialize_all_features_old(self) -> None:
        # test for deprecated version during transition period, remove after final switch
        with self.assertWarnsRegex(DeprecationWarning, "deprecated",
                                   msg="ForLoopPT does not issue warning for old serialization routines."):
            body_str = 'dt'
            dt = DummyPulseTemplate(parameter_names={'i'})

            measurements = [('a', 0, 1), ('b', 1, 1)]
            parameter_constraints = ['foo < 3']

            def make_dt(ident: str):
                self.assertEqual(body_str, ident)
                return ident

            data = dict(body=body_str,
                        loop_range=('A', 'B', 1),
                        loop_index='i',
                        identifier='meh',
                        measurements=measurements,
                        parameter_constraints=parameter_constraints)

            serializer = DummySerializer(deserialize_callback=make_dt)
            serializer.subelements['dt'] = dt

            flt = ForLoopPulseTemplate.deserialize(serializer, **data)
            self.assertEqual(flt.identifier, 'meh')
            self.assertIs(flt.body, dt)
            self.assertEqual(flt.loop_index, 'i')
            self.assertEqual(flt.loop_range.to_tuple(), ('A', 'B', 1))
            self.assertEqual(flt.measurement_declarations, measurements)
            self.assertEqual([str(c) for c in flt.parameter_constraints], parameter_constraints)


class WhileLoopPulseTemplateTest(unittest.TestCase):

    def test_parameter_names_and_declarations(self) -> None:
        condition = DummyCondition()
        body = DummyPulseTemplate()
        t = WhileLoopPulseTemplate(condition, body)
        self.assertEqual(body.parameter_names, t.parameter_names)

        body.parameter_names_ = {'foo', 't', 'bar'}
        self.assertEqual(body.parameter_names, t.parameter_names)

    @unittest.skip
    def test_is_interruptable(self) -> None:
        condition = DummyCondition()
        body = DummyPulseTemplate(is_interruptable=False)
        t = WhileLoopPulseTemplate(condition, body)
        self.assertFalse(t.is_interruptable)

        body.is_interruptable_ = True
        self.assertTrue(t.is_interruptable)

    def test_str(self) -> None:
        condition = DummyCondition()
        body = DummyPulseTemplate()
        t = WhileLoopPulseTemplate(condition, body)
        self.assertIsInstance(str(t), str)

    def test_integral(self) -> None:
        condition = DummyCondition()
        body = DummyPulseTemplate(defined_channels={'A', 'B'})
        pulse = WhileLoopPulseTemplate(condition, body)
        self.assertEqual({'A': ExpressionScalar('nan'), 'B': ExpressionScalar('nan')},
                         pulse.integral)


class WhileLoopPulseTemplateSequencingTests(unittest.TestCase):

    def test_requires_stop(self) -> None:
        condition = DummyCondition(requires_stop=False)
        conditions = {'foo_cond': condition}
        body = DummyPulseTemplate(requires_stop=False)
        t = WhileLoopPulseTemplate('foo_cond', body)
        self.assertFalse(t.requires_stop({}, conditions))

        condition.requires_stop_ = True
        self.assertTrue(t.requires_stop({}, conditions))

        body.requires_stop_ = True
        condition.requires_stop_ = False
        self.assertFalse(t.requires_stop({}, conditions))

    def test_build_sequence(self) -> None:
        condition = DummyCondition()
        body = DummyPulseTemplate()
        t = WhileLoopPulseTemplate('foo_cond', body)
        sequencer = DummySequencer()
        block = DummyInstructionBlock()
        parameters = {}
        conditions = {'foo_cond': condition}
        measurement_mapping = {'swag': 'aufdrehen'}
        channel_mapping = {}
        t.build_sequence(sequencer, parameters, conditions, measurement_mapping, channel_mapping, block)
        expected_data = dict(
            delegator=t,
            body=body,
            sequencer=sequencer,
            parameters=parameters,
            conditions=conditions,
            measurement_mapping=measurement_mapping,
            channel_mapping=channel_mapping,
            instruction_block=block
        )
        self.assertEqual(expected_data, condition.loop_call_data)
        self.assertFalse(condition.branch_call_data)
        self.assertFalse(sequencer.sequencing_stacks)

    def test_condition_missing(self) -> None:
        body = DummyPulseTemplate(requires_stop=False)
        t = WhileLoopPulseTemplate('foo_cond', body)
        sequencer = DummySequencer()
        block = DummyInstructionBlock()
        with self.assertRaises(ConditionMissingException):
            t.requires_stop({}, {})
            t.build_sequence(sequencer, {}, {}, {}, block)


class WhileLoopPulseTemplateSerializationTests(SerializableTests, unittest.TestCase):

    @property
    def class_to_test(self):
        return WhileLoopPulseTemplate

    def make_kwargs(self):
        return {
            'body': DummyPulseTemplate(),
            'condition': 'foo_cond'
        }

    def assert_equal_instance_except_id(self, lhs: WhileLoopPulseTemplate, rhs: WhileLoopPulseTemplate):
        self.assertIsInstance(lhs, WhileLoopPulseTemplate)
        self.assertIsInstance(rhs, WhileLoopPulseTemplate)
        self.assertEqual(lhs.body, rhs.body)
        self.assertEqual(lhs.condition, rhs.condition)


class WhileLoopPulseTemplateOldSerializationTests(unittest.TestCase):

    def test_get_serialization_data_old(self) -> None:
        # test for deprecated version during transition period, remove after final switch
        with self.assertWarnsRegex(DeprecationWarning, "deprecated",
                                   msg="WhileLoopPT does not issue warning for old serialization routines."):
            body = DummyPulseTemplate()
            condition_name = 'foo_cond'
            identifier = 'foo_loop'
            t = WhileLoopPulseTemplate(condition_name, body, identifier=identifier)

            serializer = DummySerializer()
            expected_data = dict(body=str(id(body)),
                                 condition=condition_name)

            data = t.get_serialization_data(serializer)
            self.assertEqual(expected_data, data)

    def test_deserialize(self) -> None:
        # test for deprecated version during transition period, remove after final switch
        with self.assertWarnsRegex(DeprecationWarning, "deprecated",
                                   msg="WhileLoopPT does not issue warning for old serialization routines."):
            data = dict(
                identifier='foo_loop',
                condition='foo_cond',
                body='bodyDummyPulse'
            )

            # prepare dependencies for deserialization
            serializer = DummySerializer()
            serializer.subelements[data['body']] = DummyPulseTemplate()

            # deserialize
            result = WhileLoopPulseTemplate.deserialize(serializer, **data)

            # compare
            self.assertIs(serializer.subelements[data['body']], result.body)
            self.assertEqual(data['condition'], result.condition)
            self.assertEqual(data['identifier'], result.identifier)


class ConditionMissingExceptionTest(unittest.TestCase):

    def test(self) -> None:
        exc = ConditionMissingException('foo')
        self.assertIsInstance(str(exc), str)


class LoopIndexNotUsedExceptionTest(unittest.TestCase):
    def str_test(self):
        self.assertEqual(str(LoopIndexNotUsedException('a', {'b', 'c'})), "The parameter a is missing in the body's parameter names: {}".format({'b', 'c'}))


if __name__ == "__main__":
    unittest.main(verbosity=2)