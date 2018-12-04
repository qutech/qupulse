import unittest
from unittest import mock
import itertools

from qupulse.pulses.mapping_pulse_template import UnnecessaryMappingException, MappingPulseTemplate,\
    AmbiguousMappingException, MappingCollisionException
from qupulse.pulses.parameters import ConstantParameter, ParameterConstraintViolation, ParameterConstraint, ParameterNotProvidedException
from qupulse.expressions import Expression
from qupulse._program._loop import Loop, MultiChannelProgram
from qupulse.pulses.sequencing import Sequencer

from tests.pulses.sequencing_dummies import DummyPulseTemplate, DummySequencer, DummyInstructionBlock, MeasurementWindowTestCase, DummyWaveform
from tests.serialization_tests import SerializableTests
from tests.serialization_dummies import DummySerializer
from tests._program.transformation_tests import TransformationStub


class MappingTemplateTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_init_exceptions(self):
        template = DummyPulseTemplate(parameter_names={'foo', 'bar'}, defined_channels={'A'}, measurement_names={'B'})
        parameter_mapping = {'foo': 't*k', 'bar': 't*l'}

        with self.assertRaises(UnnecessaryMappingException):
            MappingPulseTemplate(template, parameter_mapping=dict(**parameter_mapping, foobar='asd'))

        with self.assertRaises(UnnecessaryMappingException):
            MappingPulseTemplate(template, parameter_mapping=parameter_mapping, measurement_mapping=dict(a='b'))
        with self.assertRaises(UnnecessaryMappingException):
            MappingPulseTemplate(template, parameter_mapping=parameter_mapping, channel_mapping=dict(a='b'))

        with self.assertRaises(TypeError):
            MappingPulseTemplate(template, parameter_mapping)

        MappingPulseTemplate(template, parameter_mapping=parameter_mapping)

    def test_from_tuple_exceptions(self):
        template = DummyPulseTemplate(parameter_names={'foo', 'bar'},
                                      measurement_names={'foo', 'foobar'},
                                      defined_channels={'bar', 'foobar'})

        with self.assertRaises(ValueError):
            MappingPulseTemplate.from_tuple((template, {'A': 'B'}))
        with self.assertRaises(AmbiguousMappingException):
            MappingPulseTemplate.from_tuple((template, {'foo': 'foo'}))
        with self.assertRaises(AmbiguousMappingException):
            MappingPulseTemplate.from_tuple((template, {'bar': 'bar'}))
        with self.assertRaises(AmbiguousMappingException):
            MappingPulseTemplate.from_tuple((template, {'foobar': 'foobar'}))

        template = DummyPulseTemplate(parameter_names={'foo', 'bar'})
        with self.assertRaises(MappingCollisionException):
            MappingPulseTemplate.from_tuple((template, {'foo': '1', 'bar': 2}, {'foo': '1', 'bar': 4}))

        template = DummyPulseTemplate(defined_channels={'A'})
        with self.assertRaises(MappingCollisionException):
            MappingPulseTemplate.from_tuple((template, {'A': 'N'}, {'A': 'C'}))

        template = DummyPulseTemplate(measurement_names={'M'})
        with self.assertRaises(MappingCollisionException):
            MappingPulseTemplate.from_tuple((template, {'M': 'N'}, {'M': 'N'}))

    def test_from_tuple(self):
        template = DummyPulseTemplate(parameter_names={'foo', 'bar'},
                                      measurement_names={'m1', 'm2'},
                                      defined_channels={'c1', 'c2'})

        def test_mapping_permutations(template: DummyPulseTemplate,
                         pmap, mmap, cmap):
            direct = MappingPulseTemplate(template,
                                          parameter_mapping=pmap,
                                          measurement_mapping=mmap,
                                          channel_mapping=cmap)

            mappings = [m for m in [pmap, mmap, cmap] if m is not None]

            for current_mapping_order in itertools.permutations(mappings):
                mapper = MappingPulseTemplate.from_tuple((template, *current_mapping_order))
                self.assertEqual(mapper.measurement_mapping, direct.measurement_mapping)
                self.assertEqual(mapper.channel_mapping, direct.channel_mapping)
                self.assertEqual(mapper.parameter_mapping, direct.parameter_mapping)

        test_mapping_permutations(template, {'foo': 1, 'bar': 2},  {'m1': 'n1', 'm2': 'n2'}, {'c1': 'd1', 'c2': 'd2'})
        test_mapping_permutations(template, {'foo': 1, 'bar': 2}, {'m1': 'n1'}, {'c1': 'd1', 'c2': 'd2'})
        test_mapping_permutations(template, {'foo': 1, 'bar': 2}, None, {'c1': 'd1', 'c2': 'd2'})
        test_mapping_permutations(template, {'foo': 1, 'bar': 2}, {'m1': 'n1', 'm2': 'n2'}, {'c1': 'd1'})
        test_mapping_permutations(template, {'foo': 1, 'bar': 2}, {'m1': 'n1', 'm2': 'n2'}, None)
        test_mapping_permutations(template, None, {'m1': 'n1', 'm2': 'n2'}, {'c1': 'd1', 'c2': 'd2'})
        test_mapping_permutations(template, None, {'m1': 'n1'}, {'c1': 'd1', 'c2': 'd2'})
        test_mapping_permutations(template, None, None, {'c1': 'd1', 'c2': 'd2'})
        test_mapping_permutations(template, None, {'m1': 'n1', 'm2': 'n2'}, {'c1': 'd1'})
        test_mapping_permutations(template, None, {'m1': 'n1', 'm2': 'n2'}, None)

    def test_external_params(self):
        template = DummyPulseTemplate(parameter_names={'foo', 'bar'})
        st = MappingPulseTemplate(template, parameter_mapping={'foo': 't*k', 'bar': 't*l'})
        external_params = {'t', 'l', 'k'}
        self.assertEqual(st.parameter_names, external_params)

    def test_constrained(self):
        template = DummyPulseTemplate(parameter_names={'foo', 'bar'})
        st = MappingPulseTemplate(template, parameter_mapping={'foo': 't*k', 'bar': 't*l'}, parameter_constraints=['t < m'])
        external_params = {'t', 'l', 'k', 'm'}
        self.assertEqual(st.parameter_names, external_params)

        with self.assertRaises(ParameterConstraintViolation):
            st.map_parameters(dict(t=1, l=2, k=3, m=0))

    def test_map_parameters(self):
        template = DummyPulseTemplate(parameter_names={'foo', 'bar'})
        st = MappingPulseTemplate(template, parameter_mapping={'foo': 't*k', 'bar': 't*l'})

        parameters = {'t': ConstantParameter(3), 'k': ConstantParameter(2), 'l': ConstantParameter(7)}
        values = {'foo': 6, 'bar': 21}
        for k, v in st.map_parameters(parameters).items():
            self.assertEqual(v.get_value(), values[k])
        parameters.popitem()
        with self.assertRaises(ParameterNotProvidedException):
            st.map_parameters(parameters)

        parameters = dict(t=3, k=2, l=7)
        values = {'foo': 6, 'bar': 21}
        for k, v in st.map_parameters(parameters).items():
            self.assertEqual(v, values[k])

    def test_partial_parameter_mapping(self):
        template = DummyPulseTemplate(parameter_names={'foo', 'bar'})
        st = MappingPulseTemplate(template, parameter_mapping={'foo': 't*k'}, allow_partial_parameter_mapping=True)

        self.assertEqual(st.parameter_mapping, {'foo': 't*k', 'bar': 'bar'})

    def test_nested_mapping_avoidance(self):
        template = DummyPulseTemplate(parameter_names={'foo', 'bar'})
        st_1 = MappingPulseTemplate(template, parameter_mapping={'foo': 't*k'}, allow_partial_parameter_mapping=True)
        st_2 = MappingPulseTemplate(st_1, parameter_mapping={'bar': 't*l'}, allow_partial_parameter_mapping=True)

        self.assertIs(st_2.template, template)
        self.assertEqual(st_2.parameter_mapping, {'foo': 't*k', 'bar': 't*l'})

        st_3 = MappingPulseTemplate(template,
                                    parameter_mapping={'foo': 't*k'},
                                    allow_partial_parameter_mapping=True,
                                    identifier='käse')
        st_4 = MappingPulseTemplate(st_3, parameter_mapping={'bar': 't*l'}, allow_partial_parameter_mapping=True)
        self.assertIs(st_4.template, st_3)
        self.assertEqual(st_4.parameter_mapping, {'t': 't', 'k': 'k', 'bar': 't*l'})

    def test_parameter_names(self) -> None:
        template = DummyPulseTemplate(parameter_names={'foo'}, measurement_names={'meas1'})
        mt = MappingPulseTemplate(template, parameter_mapping={'foo': 't*k'}, parameter_constraints={'t >= m'},
                                  measurement_mapping={'meas1': 'meas2'})
        self.assertEqual({'t', 'k', 'm'}, mt.parameter_names)

    def test_get_updated_channel_mapping(self):
        template = DummyPulseTemplate(defined_channels={'foo', 'bar'})
        st = MappingPulseTemplate(template, channel_mapping={'bar': 'kneipe'})
        with self.assertRaises(KeyError):
            st.get_updated_channel_mapping(dict())
        self.assertEqual(st.get_updated_channel_mapping({'kneipe': 'meas1', 'foo': 'meas2', 'troet': 'meas3'}),
                         {'foo': 'meas2', 'bar': 'meas1'})

    def test_measurement_names(self):
        template = DummyPulseTemplate(measurement_names={'foo', 'bar'})
        st = MappingPulseTemplate(template, measurement_mapping={'foo': 'froop', 'bar': 'kneipe'})
        self.assertEqual( st.measurement_names, {'froop','kneipe'} )

    def test_defined_channels(self):
        mapping = {'asd': 'A', 'fgh': 'B'}
        template = DummyPulseTemplate(defined_channels=set(mapping.keys()))
        st = MappingPulseTemplate(template, channel_mapping=mapping)
        self.assertEqual(st.defined_channels, set(mapping.values()))

    def test_get_updated_measurement_mapping(self):
        template = DummyPulseTemplate(measurement_names={'foo', 'bar'})
        st = MappingPulseTemplate(template, measurement_mapping={'bar': 'kneipe'})
        with self.assertRaises(KeyError):
            st.get_updated_measurement_mapping(dict())
        self.assertEqual(st.get_updated_measurement_mapping({'kneipe': 'meas1', 'foo': 'meas2', 'troet': 'meas3'}),
                         {'foo': 'meas2', 'bar': 'meas1'})

    def test_integral(self) -> None:
        dummy = DummyPulseTemplate(defined_channels={'A', 'B'},
                                   parameter_names={'k', 'f', 'b'},
                                   integrals={'A': Expression('2*k'), 'other': Expression('-3.2*f+b')})
        pulse = MappingPulseTemplate(dummy, parameter_mapping={'k': 'f', 'b': 2.3}, channel_mapping={'A': 'default'},
                                     allow_partial_parameter_mapping=True)

        self.assertEqual({'default': Expression('2*f'), 'other': Expression('-3.2*f+2.3')}, pulse.integral)

    def test_mapping_namespace(self) -> None:
        template = DummyPulseTemplate(parameter_names={'foo', 'bar', 'NS(outer).NS(inner).hugo'})
        st = MappingPulseTemplate(template, mapping_namespace='NS(scope)')
        self.assertEqual({'NS(scope).foo', 'NS(scope).bar', 'NS(scope).hugo'}, st.parameter_names)

        st = MappingPulseTemplate(template, parameter_mapping={'foo': 'alpha+NS(scope).beta'}, mapping_namespace='NS(scope)')
        self.assertEqual({'alpha', 'NS(scope).beta', 'NS(scope).bar', 'NS(scope).hugo'}, st.parameter_names)


class MappingPulseTemplateSequencingTest(MeasurementWindowTestCase):

    def test_create_program(self) -> None:
        measurement_mapping = {'meas1': 'meas2'}
        parameter_mapping = {'t': 'k'}
        channel_mapping = {'B': 'default'}
        global_transformation = TransformationStub()
        to_single_waveform = {'tom', 'jerry'}

        template = DummyPulseTemplate(measurements=[('meas1', 0, 1)], measurement_names={'meas1'}, defined_channels={'B'},
                                      waveform=DummyWaveform(duration=2.0),
                                      duration=2,
                                      parameter_names={'t'})
        st = MappingPulseTemplate(template, parameter_mapping=parameter_mapping,
                                  measurement_mapping=measurement_mapping, channel_mapping=channel_mapping)

        pre_parameters = {'k': ConstantParameter(5)}
        pre_measurement_mapping = {'meas2': 'meas3'}
        pre_channel_mapping = {'default': 'A'}

        program = Loop()
        expected_inner_args = dict(parameters=st.map_parameters(pre_parameters),
                                   measurement_mapping=st.get_updated_measurement_mapping(pre_measurement_mapping),
                                   channel_mapping=st.get_updated_channel_mapping(pre_channel_mapping),
                                   to_single_waveform=to_single_waveform,
                                   global_transformation=global_transformation,
                                   parent_loop=program)

        with mock.patch.object(template, '_create_program') as inner_create_program:
            st._internal_create_program(parameters=pre_parameters,
                                        measurement_mapping=pre_measurement_mapping,
                                        channel_mapping=pre_channel_mapping,
                                        to_single_waveform=to_single_waveform,
                                        global_transformation=global_transformation,
                                        parent_loop=program)
            inner_create_program.assert_called_once_with(**expected_inner_args)

        # as we mock the inner function there shouldnt be any changes
        self.assertEqual(program, Loop())

    def test_create_program_invalid_measurement_mapping(self) -> None:
        measurement_mapping = {'meas1': 'meas2'}
        parameter_mapping = {'t': 'k'}
        channel_mapping = {'B': 'default'}

        template = DummyPulseTemplate(measurements=[('meas1', 0, 1)], measurement_names={'meas1'},
                                      defined_channels={'B'},
                                      waveform=DummyWaveform(duration=2.0),
                                      duration=2,
                                      parameter_names={'t'})
        st = MappingPulseTemplate(template, parameter_mapping=parameter_mapping,
                                  measurement_mapping=measurement_mapping, channel_mapping=channel_mapping)

        pre_parameters = {'k': ConstantParameter(5)}
        pre_measurement_mapping = {}
        pre_channel_mapping = {'default': 'A'}

        program = Loop()
        with self.assertRaises(KeyError):
            st._internal_create_program(parameters=pre_parameters,
                                        measurement_mapping=pre_measurement_mapping,
                                        channel_mapping=pre_channel_mapping,
                                        to_single_waveform=set(),
                                        global_transformation=None,
                                        parent_loop=program)

    def test_create_program_missing_params(self) -> None:
        measurement_mapping = {'meas1': 'meas2'}
        parameter_mapping = {'t': 'k'}
        channel_mapping = {'B': 'default'}

        template = DummyPulseTemplate(measurements=[('meas1', 0, 1)], measurement_names={'meas1'},
                                      defined_channels={'B'},
                                      waveform=DummyWaveform(duration=2.0),
                                      duration=2,
                                      parameter_names={'t'})
        st = MappingPulseTemplate(template, parameter_mapping=parameter_mapping,
                                  measurement_mapping=measurement_mapping, channel_mapping=channel_mapping)

        pre_parameters = {}
        pre_measurement_mapping = {'meas2': 'meas3'}
        pre_channel_mapping = {'default': 'A'}

        program = Loop()
        with self.assertRaises(ParameterNotProvidedException):
            st._internal_create_program(parameters=pre_parameters,
                                        measurement_mapping=pre_measurement_mapping,
                                        channel_mapping=pre_channel_mapping,
                                       to_single_waveform=set(),
                                       global_transformation=None,
                                        parent_loop=program)

    def test_create_program_parameter_constraint_violation(self) -> None:
        measurement_mapping = {'meas1': 'meas2'}
        parameter_mapping = {'t': 'k'}
        channel_mapping = {'B': 'default'}

        template = DummyPulseTemplate(measurements=[('meas1', 0, 1)], measurement_names={'meas1'},
                                      defined_channels={'B'},
                                      waveform=DummyWaveform(duration=2.0),
                                      duration=2,
                                      parameter_names={'t'})
        st = MappingPulseTemplate(template, parameter_mapping=parameter_mapping,
                                  measurement_mapping=measurement_mapping, channel_mapping=channel_mapping,
                                  parameter_constraints={'k > 6'})

        pre_parameters = {'k': ConstantParameter(5)}
        pre_measurement_mapping = {'meas2': 'meas3'}
        pre_channel_mapping = {'default': 'A'}

        program = Loop()
        with self.assertRaises(ParameterConstraintViolation):
            st._internal_create_program(parameters=pre_parameters,
                                        measurement_mapping=pre_measurement_mapping,
                                        channel_mapping=pre_channel_mapping,
                                      to_single_waveform=set(),
                                        global_transformation=None,
                                        parent_loop=program)

    def test_create_program_subtemplate_none(self) -> None:
        measurement_mapping = {'meas1': 'meas2'}
        parameter_mapping = {'t': 'k'}
        channel_mapping = {'B': 'default'}

        template = DummyPulseTemplate(measurements=[('meas1', 0, 1)], measurement_names={'meas1'},
                                      defined_channels={'B'},
                                      waveform=None,
                                      duration=0,
                                      parameter_names={'t'})
        st = MappingPulseTemplate(template, parameter_mapping=parameter_mapping,
                                  measurement_mapping=measurement_mapping, channel_mapping=channel_mapping)

        pre_parameters = {'k': ConstantParameter(5)}
        pre_measurement_mapping = {'meas2': 'meas3'}
        pre_channel_mapping = {'default': 'A'}

        program = Loop()
        st._internal_create_program(parameters=pre_parameters,
                                    measurement_mapping=pre_measurement_mapping,
                                    channel_mapping=pre_channel_mapping,
                                    to_single_waveform=set(),
                                    global_transformation=None,
                                    parent_loop=program)

        self.assertEqual(1, len(template.create_program_calls))
        self.assertEqual((st.map_parameters(pre_parameters),
                          st.get_updated_measurement_mapping(pre_measurement_mapping),
                          st.get_updated_channel_mapping(pre_channel_mapping),
                          program),
                         template.create_program_calls[-1])

        self.assertEqual(1, program.repetition_count)
        self.assertEqual(0, len(program.children))
        self.assertIsNone(program._measurements)

        # ensure same result as from Sequencer
        sequencer = Sequencer()
        sequencer.push(st, parameters=pre_parameters, conditions={}, window_mapping=pre_measurement_mapping,
                       channel_mapping=pre_channel_mapping)
        block = sequencer.build()
        program_old = MultiChannelProgram(block, channels={'A'}).programs[frozenset({'A'})]
        self.assertEqual(program_old, program)


class MappingPulseTemplateOldSequencingTests(unittest.TestCase):

    def test_build_sequence(self):
        measurement_mapping = {'meas1': 'meas2'}
        parameter_mapping = {'t': 'k'}

        template = DummyPulseTemplate(measurement_names=set(measurement_mapping.keys()),
                                      parameter_names=set(parameter_mapping.keys()))
        st = MappingPulseTemplate(template, parameter_mapping=parameter_mapping, measurement_mapping=measurement_mapping)
        sequencer = DummySequencer()
        block = DummyInstructionBlock()
        pre_parameters = {'k': ConstantParameter(5)}
        pre_measurement_mapping = {'meas2': 'meas3'}
        pre_channel_mapping = {'default': 'A'}
        conditions = dict(a=True)
        st.build_sequence(sequencer, pre_parameters, conditions, pre_measurement_mapping, pre_channel_mapping, block)

        self.assertEqual(template.build_sequence_calls, 1)
        forwarded_args = template.build_sequence_arguments[0]
        self.assertEqual(forwarded_args[0], sequencer)
        self.assertEqual(forwarded_args[1], st.map_parameters(pre_parameters))
        self.assertEqual(forwarded_args[2], conditions)
        self.assertEqual(forwarded_args[3],
                         st.get_updated_measurement_mapping(pre_measurement_mapping))
        self.assertEqual(forwarded_args[4],
                         st.get_updated_channel_mapping(pre_channel_mapping))
        self.assertEqual(forwarded_args[5], block)

    @unittest.skip("Extend of dummy template for argument checking needed.")
    def test_requires_stop(self):
        pass


class PulseTemplateParameterMappingExceptionsTests(unittest.TestCase):

    def test_unnecessary_mapping_exception_str(self) -> None:
        dummy = DummyPulseTemplate()
        exception = UnnecessaryMappingException(dummy, 'foo')
        self.assertIsInstance(str(exception), str)


class MappingPulseTemplateSerializationTests(SerializableTests, unittest.TestCase):

    @property
    def class_to_test(self):
        return MappingPulseTemplate

    def make_kwargs(self):
        return {
            'template': DummyPulseTemplate(defined_channels={'foo'},
                                           measurement_names={'meas'},
                                           parameter_names={'hugo', 'herbert', 'ilse'}),
            'parameter_mapping': {'hugo': Expression('2*k+c'), 'herbert': Expression('c-1.5'), 'ilse': Expression('ilse')},
            'measurement_mapping': {'meas': 'seam'},
            'channel_mapping': {'foo': 'default_channel'},
            'parameter_constraints': [str(ParameterConstraint('c > 0'))]
        }

    def make_instance(self, identifier=None, registry=None):
        kwargs = self.make_kwargs()
        return self.class_to_test(identifier=identifier, **kwargs, allow_partial_parameter_mapping=True, registry=registry)

    def assert_equal_instance_except_id(self, lhs: MappingPulseTemplate, rhs: MappingPulseTemplate):
        self.assertIsInstance(lhs, MappingPulseTemplate)
        self.assertIsInstance(rhs, MappingPulseTemplate)
        self.assertEqual(lhs.template, rhs.template)
        self.assertEqual(lhs.parameter_constraints, rhs.parameter_constraints)
        self.assertEqual(lhs.channel_mapping, rhs.channel_mapping)
        self.assertEqual(lhs.measurement_mapping, rhs.measurement_mapping)
        self.assertEqual(lhs.parameter_mapping, rhs.parameter_mapping)


class MappingPulseTemplateOldSerializationTests(unittest.TestCase):

    def test_get_serialization_data(self) -> None:
        # test for deprecated version during transition period, remove after final switch
        with self.assertWarnsRegex(DeprecationWarning, "deprecated",
                                   msg="SequencePT does not issue warning for old serialization routines."):
            dummy_pt = DummyPulseTemplate(defined_channels={'foo'},
                                          measurement_names={'meas'},
                                          parameter_names={'hugo', 'herbert', 'ilse'})
            mpt = MappingPulseTemplate(
                template=dummy_pt,
                parameter_mapping={'hugo': Expression('2*k+c'), 'herbert': Expression('c-1.5'), 'ilse': Expression('ilse')},
                measurement_mapping={'meas': 'seam'},
                channel_mapping={'foo': 'default_channel'},
                parameter_constraints=[str(ParameterConstraint('c > 0'))]
            )
            serializer = DummySerializer()
            expected_data = {
                'template': serializer.dictify(dummy_pt),
                'parameter_mapping': {'hugo': str(Expression('2*k+c')), 'herbert': str(Expression('c-1.5')),
                                      'ilse': str(Expression('ilse'))},
                'measurement_mapping': {'meas': 'seam'},
                'channel_mapping': {'foo': 'default_channel'},
                'parameter_constraints': [str(ParameterConstraint('c > 0'))]
            }
            data = mpt.get_serialization_data(serializer=serializer)
            self.assertEqual(expected_data, data)

    def test_deserialize(self) -> None:
        # test for deprecated version during transition period, remove after final switch
        with self.assertWarnsRegex(DeprecationWarning, "deprecated",
                                   msg="SequencePT does not issue warning for old serialization routines."):
            dummy_pt = DummyPulseTemplate(defined_channels={'foo'},
                                          measurement_names={'meas'},
                                          parameter_names={'hugo', 'herbert', 'ilse'})
            serializer = DummySerializer()
            data = {
                'template': serializer.dictify(dummy_pt),
                'parameter_mapping': {'hugo': str(Expression('2*k+c')), 'herbert': str(Expression('c-1.5')),
                                      'ilse': str(Expression('ilse'))},
                'measurement_mapping': {'meas': 'seam'},
                'channel_mapping': {'foo': 'default_channel'},
                'parameter_constraints': [str(ParameterConstraint('c > 0'))]
            }
            deserialized = MappingPulseTemplate.deserialize(serializer=serializer, **data)

            self.assertIsInstance(deserialized, MappingPulseTemplate)
            self.assertEqual(data['parameter_mapping'], deserialized.parameter_mapping)
            self.assertEqual(data['channel_mapping'], deserialized.channel_mapping)
            self.assertEqual(data['measurement_mapping'], deserialized.measurement_mapping)
            self.assertEqual(data['parameter_constraints'], [str(pc) for pc in deserialized.parameter_constraints])
            self.assertIs(deserialized.template, dummy_pt)