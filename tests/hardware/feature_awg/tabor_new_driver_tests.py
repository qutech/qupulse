import unittest

try:
    import tabor_control
except ImportError as err:
    raise unittest.SkipTest("tabor_control not present") from err

from qupulse.hardware.feature_awg.tabor import with_configuration_guard


class ConfigurationGuardTest(unittest.TestCase):
    class DummyChannelPair:
        def __init__(self, test_obj: unittest.TestCase):
            self.test_obj = test_obj
            self._configuration_guard_count = 0
            self.is_in_config_mode = False

        def _enter_config_mode(self):
            self.test_obj.assertFalse(self.is_in_config_mode)
            self.test_obj.assertEqual(self._configuration_guard_count, 0)
            self.is_in_config_mode = True

        def _exit_config_mode(self):
            self.test_obj.assertTrue(self.is_in_config_mode)
            self.test_obj.assertEqual(self._configuration_guard_count, 0)
            self.is_in_config_mode = False

        @with_configuration_guard
        def guarded_method(self, counter=5, throw=False):
            self.test_obj.assertTrue(self.is_in_config_mode)
            if counter > 0:
                return self.guarded_method(counter - 1, throw) + 1
            if throw:
                raise RuntimeError()
            return 0

    def test_config_guard(self):
        channel_pair = ConfigurationGuardTest.DummyChannelPair(self)

        for i in range(5):
            self.assertEqual(channel_pair.guarded_method(i), i)

        with self.assertRaises(RuntimeError):
            channel_pair.guarded_method(1, True)

        self.assertFalse(channel_pair.is_in_config_mode)


