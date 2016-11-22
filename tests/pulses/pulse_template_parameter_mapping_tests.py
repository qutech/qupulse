import unittest

from qctoolkit.pulses.pulse_template import SubTemplate
from qctoolkit.pulses.pulse_template_parameter_mapping import PulseTemplateParameterMapping, MissingMappingException,\
    UnnecessaryMappingException, MissingParameterDeclarationException, get_measurement_name_mappings
from qctoolkit.expressions import Expression
from qctoolkit.pulses.parameters import MappedParameter, ParameterNotProvidedException, ConstantParameter

from tests.pulses.sequencing_dummies import DummyPulseTemplate


class PulseTemplateParameterMappingTests(unittest.TestCase):

    def __init__(self, methodName) -> None:
        super().__init__(methodName=methodName)

    def test_empty_init(self) -> None:
        map = PulseTemplateParameterMapping()
        self.assertEqual(set(), map.external_parameters)

    def test_set_external_parameters(self) -> None:
        map = PulseTemplateParameterMapping({'foo'})
        self.assertEqual({'foo'}, map.external_parameters)
        map.set_external_parameters(None)
        self.assertEqual({'foo'}, map.external_parameters)
        map.set_external_parameters({'bar'})
        self.assertEqual({'bar'}, map.external_parameters)
        map.set_external_parameters(set())
        self.assertEqual(set(), map.external_parameters)

    def test_add_unnecessary_mapping(self) -> None:
        map = PulseTemplateParameterMapping()
        dummy = DummyPulseTemplate(parameter_names={'foo'})
        with self.assertRaises(UnnecessaryMappingException):
            map.add(dummy, 'bar', '2')

    def test_add_missing_external_parameter(self) -> None:
        map = PulseTemplateParameterMapping()
        dummy = DummyPulseTemplate(parameter_names={'foo'})
        with self.assertRaises(MissingParameterDeclarationException):
            map.add(dummy, 'foo', 'bar')

    def test_add(self) -> None:
        map = PulseTemplateParameterMapping({'bar'})
        dummy1 = DummyPulseTemplate(parameter_names={'foo', 'hugo'})
        dummy2 = DummyPulseTemplate(parameter_names={'grr'})
        map.add(dummy1, 'foo', '4*bar')
        map.add(dummy2, 'grr', Expression('bar ** 2'))
        map.add(dummy1, 'hugo', '3')
        map.add(dummy2, 'grr', Expression('sin(bar)'))
        self.assertEqual(dict(foo=Expression('4*bar'), hugo=Expression('3')), map.get_template_map(dummy1))
        self.assertEqual(dict(grr=Expression('sin(bar)')), map.get_template_map(dummy2))

    def test_get_template_map_no_key(self) -> None:
        map = PulseTemplateParameterMapping()
        dummy = DummyPulseTemplate()
        self.assertEqual(dict(), map.get_template_map(dummy))

    def test_is_template_mapped(self) -> None:
        map = PulseTemplateParameterMapping({'bar'})
        dummy1 = DummyPulseTemplate(parameter_names={'foo', 'hugo'})
        dummy2 = DummyPulseTemplate(parameter_names={'grr'})
        map.add(dummy1, 'foo', '4*bar')
        self.assertFalse(map.is_template_mapped(dummy1))
        map.add(dummy1, 'hugo', 'bar + 1')
        self.assertTrue(map.is_template_mapped(dummy1))
        self.assertFalse(map.is_template_mapped(dummy2))

    def test_get_remaining_mappings(self) -> None:
        map = PulseTemplateParameterMapping({'bar', 'barbar'})
        dummy = DummyPulseTemplate(parameter_names={'foo', 'hugo'})
        self.assertEqual({'foo', 'hugo'}, map.get_remaining_mappings(dummy))
        map.add(dummy, 'hugo', '4*bar')
        self.assertEqual({'foo'}, map.get_remaining_mappings(dummy))
        map.add(dummy, 'foo', Expression('barbar'))
        self.assertEqual(set(), map.get_remaining_mappings(dummy))

    def test_map_parameters_not_provided(self) -> None:
        map = PulseTemplateParameterMapping({'bar', 'barbar'})
        dummy = DummyPulseTemplate(parameter_names={'foo', 'hugo'})
        map.add(dummy, 'hugo', '4*bar')
        map.add(dummy, 'foo', Expression('barbar'))
        with self.assertRaises(ParameterNotProvidedException):
            map.map_parameters(dummy, dict(bar=ConstantParameter(3)))

    def test_map_parameters(self) -> None:
        map = PulseTemplateParameterMapping({'bar', 'barbar'})
        dummy = DummyPulseTemplate(parameter_names={'foo', 'hugo'})
        map.add(dummy, 'hugo', '4*bar')
        map.add(dummy, 'foo', Expression('barbar'))
        mapped = map.map_parameters(dummy, dict(bar=ConstantParameter(3), barbar=ConstantParameter(5)))
        self.assertEqual(dict(
            hugo=MappedParameter(Expression('4*bar'), dict(bar=ConstantParameter(3))),
            foo=MappedParameter(Expression('barbar'), dict(barbar=ConstantParameter(5)))
        ), mapped)


class GetMeasurementNameMappingsTest(unittest.TestCase):
    def setUp(self):
        self.template = DummyPulseTemplate()

    def test_no_mapping(self):
        template = DummyPulseTemplate()

        mappings = get_measurement_name_mappings([SubTemplate(template, dict())])
        self.assertEqual(mappings,[dict()])

    def test_mapping(self):
        template = DummyPulseTemplate()
        template.measurement_names_ = set(['foo', 'bar'])

        mapping_arg={'foo': 'hendrix', 'bar': 'shoe'}
        mappings = get_measurement_name_mappings([SubTemplate(template,dict(),measurement_mapping=mapping_arg)])

        self.assertEqual(mappings,[mapping_arg])

    def test_missing_mapping(self):
        template = DummyPulseTemplate()
        template.measurement_names_ = set(['foo', 'bar'])

        mapping_arg = {'foo': 'hendrix'}
        mappings = get_measurement_name_mappings([SubTemplate(template, dict(), measurement_mapping=mapping_arg)])

        self.assertEqual(mappings, [dict(**mapping_arg,bar='bar')])

    def test_unnecessary_mapping(self):
        template = DummyPulseTemplate()
        template.measurement_names_ = set(['foo', 'bar'])

        mapping_arg = {'foo': 'hendrix', 'my': 'oh_my'}
        with self.assertRaises(UnnecessaryMappingException):
            get_measurement_name_mappings([SubTemplate(template, dict(), measurement_mapping=mapping_arg)])

    def test_multiple_arguments(self):
        template1 = DummyPulseTemplate()
        template1.measurement_names_ = set(['foo', 'bar'])

        template2 = DummyPulseTemplate()
        template2.measurement_names_ = set(['hurr', 'durr'])

        mappings = get_measurement_name_mappings([SubTemplate(template1,dict(),measurement_mapping=dict()),
                                                 SubTemplate(template1,dict(),measurement_mapping={'foo': 'bar'}),
                                                 SubTemplate(template2, dict(), measurement_mapping={'hurr': 'bar','durr': 'bar'})])
        self.assertEqual(mappings,[dict(foo='foo',bar='bar'),dict(foo='bar',bar='bar'),{'hurr': 'bar','durr': 'bar'}])

        with self.assertRaises(UnnecessaryMappingException):
            get_measurement_name_mappings([SubTemplate(template1, dict(), measurement_mapping=dict()),
                                          SubTemplate(template1, dict(),
                                                      measurement_mapping={'foo': 'bar'}),
                                          SubTemplate(template2, dict(),
                                                      measurement_mapping={'hurr': 'bar', 'durr': 'bar', 'hopfen': 'malz'})])

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
