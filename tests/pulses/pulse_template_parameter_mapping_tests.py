import unittest
import warnings

from qupulse.serialization import Serializer
from tests.pulses.sequencing_dummies import DummyPulseTemplate
from tests.serialization_dummies import DummyStorageBackend


class TestPulseTemplateParameterMappingFileTests(unittest.TestCase):

    # ensure that a MappingPulseTemplate imported from pulse_template_parameter_mapping serializes as from mapping_pulse_template
    def test_pulse_template_parameter_include(self) -> None:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('ignore', DeprecationWarning)
            from qupulse.pulses.pulse_template_parameter_mapping import MappingPulseTemplate
            dummy_t = DummyPulseTemplate()
            map_t = MappingPulseTemplate(dummy_t)
            type_str = map_t.get_type_identifier()
            self.assertEqual("qupulse.pulses.mapping_pulse_template.MappingPulseTemplate", type_str)

