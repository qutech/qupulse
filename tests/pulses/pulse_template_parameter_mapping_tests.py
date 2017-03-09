import unittest

from qctoolkit.pulses.pulse_template_parameter_mapping import MissingMappingException,\
    UnnecessaryMappingException, MissingParameterDeclarationException, MappingTemplate
from qctoolkit.expressions import Expression
from qctoolkit.pulses.parameters import ParameterNotProvidedException
from qctoolkit.pulses.parameters import ConstantParameter

from tests.pulses.sequencing_dummies import DummyPulseTemplate, DummySequencer, DummyInstructionBlock


class MappingTemplateTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_init(self):
        template = DummyPulseTemplate(parameter_names={'foo', 'bar'})
        parameter_mapping = {'foo': 't*k', 'bar': 't*l'}

        with self.assertRaises(MissingMappingException):
            MappingTemplate(template, parameter_mapping={})
        with self.assertRaises(MissingMappingException):
            MappingTemplate(template, parameter_mapping={'bar': 'kneipe'})
        with self.assertRaises(UnnecessaryMappingException):
            MappingTemplate(template, dict(**parameter_mapping, foobar='asd'))
        MappingTemplate(template, parameter_mapping=parameter_mapping)

        with self.assertRaises(UnnecessaryMappingException):
            MappingTemplate(template, parameter_mapping, measurement_mapping=dict(a='b'))
        with self.assertRaises(UnnecessaryMappingException):
            MappingTemplate(template, parameter_mapping, channel_mapping=dict(a='b'))

    def test_external_params(self):
        template = DummyPulseTemplate(parameter_names={'foo', 'bar'})
        st = MappingTemplate(template, parameter_mapping={'foo': 't*k', 'bar': 't*l'})
        external_params = {'t', 'l', 'k'}
        self.assertEqual(st.parameter_names, external_params)

    def test_map_parameters(self):
        template = DummyPulseTemplate(parameter_names={'foo', 'bar'})
        st = MappingTemplate(template, parameter_mapping={'foo': 't*k', 'bar': 't*l'})

        parameters = {'t': ConstantParameter(3), 'k': ConstantParameter(2), 'l': ConstantParameter(7)}
        values = {'foo': 6, 'bar': 21}
        for k, v in st.map_parameters(parameters).items():
            self.assertEqual(v.get_value(), values[k])
        parameters.popitem()
        with self.assertRaises(ParameterNotProvidedException):
            st.map_parameters(parameters)

    def test_get_updated_channel_mapping(self):
        template = DummyPulseTemplate(defined_channels={'foo', 'bar'})
        st = MappingTemplate(template, {}, channel_mapping={'bar': 'kneipe'})
        with self.assertRaises(KeyError):
            st.get_updated_channel_mapping(dict())
        self.assertEqual(st.get_updated_channel_mapping({'kneipe': 'meas1', 'foo': 'meas2', 'troet': 'meas3'}),
                         {'foo': 'meas2', 'bar': 'meas1'})

    def test_measurement_names(self):
        template = DummyPulseTemplate(measurement_names={'foo', 'bar'})
        st = MappingTemplate(template, {}, measurement_mapping={'foo': 'froop', 'bar': 'kneipe'})
        self.assertEqual( st.measurement_names, {'froop','kneipe'} )

    def test_defined_channels(self):
        mapping = {'asd': 'A', 'fgh': 'B'}
        template = DummyPulseTemplate(defined_channels=set(mapping.keys()))
        st = MappingTemplate(template, {}, channel_mapping=mapping)
        self.assertEqual(st.defined_channels, set(mapping.values()))

    def test_get_updated_measurement_mapping(self):
        template = DummyPulseTemplate(measurement_names={'foo', 'bar'})
        st = MappingTemplate(template, {}, measurement_mapping={'bar': 'kneipe'})
        with self.assertRaises(KeyError):
            st.get_updated_measurement_mapping(dict())
        self.assertEqual(st.get_updated_measurement_mapping({'kneipe': 'meas1', 'foo': 'meas2', 'troet': 'meas3'}),
                         {'foo': 'meas2', 'bar': 'meas1'})

    def test_build_sequence(self):
        measurement_mapping = {'meas1': 'meas2'}
        parameter_mapping = {'t': 'k'}

        template = DummyPulseTemplate(measurement_names=set(measurement_mapping.keys()),
                                      parameter_names=set(parameter_mapping.keys()))
        st = MappingTemplate(template, parameter_mapping, measurement_mapping=measurement_mapping)
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
