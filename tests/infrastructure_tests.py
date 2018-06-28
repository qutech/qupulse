import unittest

from qctoolkit.infrastructure import ParameterLibrary, ParameterDict

from tests.pulses.sequencing_dummies import DummyPulseTemplate


class ParameterLibraryTests(unittest.TestCase):

    def setUp(self) -> None:
        self.high_level_params = {
            'global': {'foo': 0.32, 'bar': 3.156, 'ilse': -1.2365, 'hugo': -15.236},
            'test_pulse': {'foo': -2.4},
            'tast_pulse': {'foo': 0, 'ilse': 0}
        }
        self.intermediate_level_params = {
            'global': {'foo': -1.176, 'hugo': 0.151}
        }
        self.low_level_params = {
            'test_pulse': {'bar': -2.75},
            'tast_pulse': {'foo': 0.12}
        }
        self.param_sources = [self.high_level_params, self.intermediate_level_params, self.low_level_params]

    def test_get_parameters_1(self) -> None:
        pt = DummyPulseTemplate(parameter_names={'foo', 'bar', 'ilse'}, identifier='test_pulse')
        expected_params = {
            'foo': self.intermediate_level_params['global']['foo'],
            'bar': self.low_level_params[pt.identifier]['bar'],
            'ilse': self.high_level_params['global']['ilse'],
            'hugo': self.intermediate_level_params['global']['hugo']
        }

        composer = ParameterLibrary(self.param_sources)
        params = composer.get_parameters(pt)

        self.assertEqual(expected_params, params)

    def test_get_parameters_2(self) -> None:
        pt = DummyPulseTemplate(parameter_names={'foo', 'hugo', 'ilse'}, identifier='tast_pulse')
        expected_params = {
            'foo': self.low_level_params[pt.identifier]['foo'],
            'bar': self.high_level_params['global']['bar'],
            'hugo': self.intermediate_level_params['global']['hugo'],
            'ilse': self.high_level_params[pt.identifier]['ilse']
        }

        composer = ParameterLibrary(self.param_sources)
        params = composer.get_parameters(pt)

        self.assertEqual(expected_params, params)

    def test_get_parameters_with_local_subst(self) -> None:
        pt = DummyPulseTemplate(parameter_names={'foo', 'hugo', 'ilse'}, identifier='tast_pulse')
        local_param_subst = {
            'ilse': -12.5,
            'hugo': 7.25
        }

        expected_params = {
            'foo': self.low_level_params[pt.identifier]['foo'],
            'bar': self.high_level_params['global']['bar'],
            'ilse': local_param_subst['ilse'],
            'hugo': local_param_subst['hugo']
        }

        composer = ParameterLibrary(self.param_sources)
        params = composer.get_parameters(pt, local_param_subst)

        self.assertEqual(expected_params, params)

    def test_no_pulse_params(self) -> None:
        pt = DummyPulseTemplate(parameter_names={'foo', 'bar', 'hugo'}, identifier='unknown_pulse')
        expected_params = {
            'foo': self.intermediate_level_params['global']['foo'],
            'bar': self.high_level_params['global']['bar'],
            'ilse': self.high_level_params['global']['ilse'],
            'hugo': self.intermediate_level_params['global']['hugo']
        }

        composer = ParameterLibrary(self.param_sources)
        params = composer.get_parameters(pt)

        self.assertEqual(expected_params, params)

    def test_no_puls_id(self) -> None:
        pt = DummyPulseTemplate(parameter_names={'foo', 'bar', 'hugo'})
        expected_params = {
            'foo': self.intermediate_level_params['global']['foo'],
            'bar': self.high_level_params['global']['bar'],
            'ilse': self.high_level_params['global']['ilse'],
            'hugo': self.intermediate_level_params['global']['hugo']
        }

        composer = ParameterLibrary(self.param_sources)
        params = composer.get_parameters(pt)

        self.assertEqual(expected_params, params)

    def test_no_pulse_params_subst(self) -> None:
        pt = DummyPulseTemplate(parameter_names={'foo', 'bar', 'hugo'}, identifier='unknown_pulse')
        local_param_subst = {
            'ilse': -12.5,
            'hugo': 7.25
        }
        expected_params = {
            'foo': self.intermediate_level_params['global']['foo'],
            'bar': self.high_level_params['global']['bar'],
            'ilse': local_param_subst['ilse'],
            'hugo': local_param_subst['hugo']
        }

        composer = ParameterLibrary(self.param_sources)
        params = composer.get_parameters(pt, local_param_subst)

        self.assertEqual(expected_params, params)

    def test_global_or_pulse_params(self) -> None:
        high_level_params = {
            'test_pulse': {'foo': -2.4},
            'tast_pulse': {'foo': 0, 'ilse': 0}
        }
        low_level_params = {
            'test_pulse': {'bar': -2.75},
            'tast_pulse': {'foo': 0.12}
        }
        pt = DummyPulseTemplate(parameter_names={'foo', 'bar', 'hugo'}, identifier='unknown_pulse')
        local_param_subst = {
            'ilse': -12.5,
            'hugo': 7.25
        }
        expected_params = {
            'ilse': local_param_subst['ilse'],
            'hugo': local_param_subst['hugo']
        }

        composer = ParameterLibrary([high_level_params, low_level_params])
        params = composer.get_parameters(pt, local_param_subst)

        self.assertEqual(expected_params, params)
