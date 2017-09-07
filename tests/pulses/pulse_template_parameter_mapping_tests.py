import unittest
import itertools

from qctoolkit.pulses.pulse_template_parameter_mapping import MissingMappingException,\
    UnnecessaryMappingException, MissingParameterDeclarationException, MappingPulseTemplate,\
    AmbiguousMappingException, MappingCollisionException
from qctoolkit.expressions import Expression
from qctoolkit.pulses.parameters import ParameterNotProvidedException
from qctoolkit.pulses.parameters import ConstantParameter, ParameterConstraintViolation

from tests.pulses.sequencing_dummies import DummyPulseTemplate, DummySequencer, DummyInstructionBlock


class MappingTemplateTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_init_exceptions(self):
        template = DummyPulseTemplate(parameter_names={'foo', 'bar'}, defined_channels={'A'}, measurement_names={'B'})
        parameter_mapping = {'foo': 't*k', 'bar': 't*l'}

        with self.assertRaises(MissingMappingException):
            MappingPulseTemplate(template, parameter_mapping={})
        with self.assertRaises(MissingMappingException):
            MappingPulseTemplate(template, parameter_mapping={'bar': 'kneipe'})
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

    def test_missing_parameter_declaration_exception_str(self) -> None:
        dummy = DummyPulseTemplate()
        exception = MissingParameterDeclarationException(dummy, 'foo')
        self.assertIsInstance(str(exception), str)

    def test_missing_mapping_exception_str(self) -> None:
        dummy = DummyPulseTemplate()
        exception = MissingMappingException(dummy, 'foo')
        self.assertIsInstance(str(exception), str)

    def test_unnecessary_mapping_exception_str(self) -> None:
        dummy = DummyPulseTemplate()
        exception = UnnecessaryMappingException(dummy, 'foo')
        self.assertIsInstance(str(exception), str)
