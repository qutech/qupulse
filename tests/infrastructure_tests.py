import unittest

from qctoolkit.infrastructure import ParameterDictComposer, ParameterDict

from tests.pulses.sequencing_dummies import DummyPulseTemplate


class ParameterDictComposerTests(unittest.TestCase):

    def setUp(self) -> None:
        high_level_params = {
            'global': {'foo': 0.32, 'hugo': -15.236, 'bar': 3.156, 'ilse': -1.2365},
            'test_pulse': {'foo': -2.4},
            'tast_pulse': {'foo': 0, 'ilse': 0}
        }
        intermediate_level_params = {
            'global': {'foo': -1.176, 'hugo': 0.151}
        }
        low_level_params = {
            'test_pulse': {'bar': -2.75},
            'tast_pulse': {'foo': 0.12}
        }
        self.param_sources = [high_level_params, intermediate_level_params, low_level_params]

    def test_get_parameters_1(self) -> None:
        pt = DummyPulseTemplate(parameter_names={'foo', 'bar', 'ilse'}, identifier='test_pulse')
        expected_params = {
            'foo': -1.176,
            'bar': -2.75,
            'ilse': -1.2365
        }

        composer = ParameterDictComposer(self.param_sources)
        params = composer.get_parameters(pt)

        self.assertEqual(expected_params, params)

    def test_get_parameters_2(self) -> None:
        pt = DummyPulseTemplate(parameter_names={'foo', 'hugo', 'ilse'}, identifier='tast_pulse')
        expected_params = {
            'foo': 0.12,
            'hugo': 0.151,
            'ilse': 0
        }

        composer = ParameterDictComposer(self.param_sources)
        params = composer.get_parameters(pt)

        self.assertEqual(expected_params, params)