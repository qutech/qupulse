import unittest
from unittest import mock
from collections import OrderedDict

import numpy as np

try:
    import pytest
except ImportError:
    pytest = None

if pytest:
    zhinst = pytest.importorskip("zhinst")

    try:
        import zhinst.core as zhinst_core
    except ImportError:
        import zhinst.ziPython as zhinst_core
else:
    try:
        try:
            import zhinst.core as zhinst_core
        except ImportError:
            import zhinst.ziPython as zhinst_core
    except ImportError as err:
        raise unittest.SkipTest("zhinst not present") from err

from qupulse.utils.types import TimeType
from qupulse._program._loop import Loop
from tests.pulses.sequencing_dummies import DummyWaveform
from qupulse.hardware.awgs.zihdawg import HDAWGChannelGroup, HDAWGRepresentation, HDAWGValueError, UserRegister,\
    ELFManager, HDAWGChannelGrouping, SingleDeviceChannelGroup


class HDAWGRepresentationTests(unittest.TestCase):
    def test_init(self):
        """We do not test anything lab one related"""
        device_serial = 'dev6ä6ä6'
        device_interface = 'telepathy'
        data_server_addr = 'asd'
        data_server_port = 42
        api_level_number = 23
        channel_grouping = HDAWGChannelGrouping.CHAN_GROUP_1x8

        with \
                mock.patch('zhinst.utils.api_server_version_check') as mock_version_check,\
                mock.patch.object(zhinst_core, 'ziDAQServer') as mock_daq_server, \
                mock.patch('qupulse.hardware.awgs.zihdawg.HDAWGRepresentation._initialize') as mock_init, \
                mock.patch('qupulse.hardware.awgs.zihdawg.HDAWGRepresentation.channel_grouping', new_callable=mock.PropertyMock) as mock_grouping, \
                mock.patch('qupulse.hardware.awgs.zihdawg.SingleDeviceChannelGroup') as mock_channel_pair,\
                mock.patch('zhinst.utils.disable_everything') as mock_reset,\
                mock.patch('pathlib.Path') as mock_path:

            representation = HDAWGRepresentation(device_serial,
                                                 device_interface,
                                                 data_server_addr, data_server_port, api_level_number,
                                                 False, 1.3, grouping=channel_grouping)

            mock_daq_server.return_value.awgModule.return_value.getString.assert_called_once_with('directory')
            module_dir = mock_daq_server.return_value.awgModule.return_value.getString.return_value
            mock_path.assert_called_once_with(module_dir, 'awg', 'waves')

            self.assertIs(representation.api_session, mock_daq_server.return_value)
            mock_daq_server.assert_called_once_with(data_server_addr, data_server_port, api_level_number)

            mock_version_check.assert_called_once_with(representation.api_session)
            representation.api_session.connectDevice.assert_called_once_with(device_serial, device_interface)
            self.assertEqual(device_serial, representation.serial)

            mock_grouping.assert_called_once_with(channel_grouping)

            mock_reset.assert_not_called()
            mock_init.assert_called_once_with()

            group_calls = [mock.call(0, 2, identifier=str(device_serial) + '_AB', timeout=1.3),
                           mock.call(1, 2, identifier=str(device_serial) + '_CD', timeout=1.3),
                           mock.call(2, 2, identifier=str(device_serial) + '_EF', timeout=1.3),
                           mock.call(3, 2, identifier=str(device_serial) + '_GH', timeout=1.3),
                           mock.call(0, 4, identifier=str(device_serial) + '_ABCD', timeout=1.3),
                           mock.call(1, 4, identifier=str(device_serial) + '_EFGH', timeout=1.3),
                           mock.call(0, 8, identifier=str(device_serial) + '_ABCDEFGH', timeout=1.3)]
            for c1, c2 in zip(group_calls, mock_channel_pair.call_args_list):
                self.assertEqual(c1, c2)

            self.assertIs(representation.channel_pair_AB, mock_channel_pair.return_value)
            self.assertIs(representation.channel_pair_CD, mock_channel_pair.return_value)
            self.assertIs(representation.channel_pair_EF, mock_channel_pair.return_value)
            self.assertIs(representation.channel_pair_GH, mock_channel_pair.return_value)

            mock_version_check.reset_mock()
            mock_daq_server.reset_mock()
            mock_init.reset_mock()
            mock_channel_pair.reset_mock()
            mock_reset.reset_mock()

            representation = HDAWGRepresentation(device_serial,
                                                 device_interface,
                                                 data_server_addr, data_server_port, api_level_number, True)

            self.assertIs(representation.api_session, mock_daq_server.return_value)
            mock_daq_server.assert_called_once_with(data_server_addr, data_server_port, api_level_number)

            mock_version_check.assert_called_once_with(representation.api_session)
            representation.api_session.connectDevice.assert_called_once_with(device_serial, device_interface)
            self.assertEqual(device_serial, representation.serial)

            mock_reset.assert_called_once_with(representation.api_session, representation.serial)
            mock_init.assert_called_once_with()

            group_calls = [mock.call(0, 2, identifier=str(device_serial) + '_AB', timeout=20),
                           mock.call(1, 2, identifier=str(device_serial) + '_CD', timeout=20),
                           mock.call(2, 2, identifier=str(device_serial) + '_EF', timeout=20),
                           mock.call(3, 2, identifier=str(device_serial) + '_GH', timeout=20),
                           mock.call(0, 4, identifier=str(device_serial) + '_ABCD', timeout=20),
                           mock.call(1, 4, identifier=str(device_serial) + '_EFGH', timeout=20),
                           mock.call(0, 8, identifier=str(device_serial) + '_ABCDEFGH', timeout=20)]
            self.assertEqual(group_calls, mock_channel_pair.call_args_list)

            self.assertIs(representation.channel_pair_AB, mock_channel_pair.return_value)
            self.assertIs(representation.channel_pair_CD, mock_channel_pair.return_value)
            self.assertIs(representation.channel_pair_EF, mock_channel_pair.return_value)
            self.assertIs(representation.channel_pair_GH, mock_channel_pair.return_value)


class HDAWGChannelGroupTests(unittest.TestCase):
    def test_init(self):
        with mock.patch('weakref.proxy') as proxy_mock:
            mock_device = mock.Mock()

            channels = (3, 4)
            awg_group_idx = 1

            channel_pair = SingleDeviceChannelGroup(awg_group_idx, 2, 'foo', 3.4)

            self.assertEqual(channel_pair.timeout, 3.4)
            self.assertEqual(channel_pair._channels(), channels)
            self.assertEqual(channel_pair.awg_group_index, awg_group_idx)
            self.assertEqual(channel_pair.num_channels, 2)
            self.assertEqual(channel_pair.num_markers, 4)

            self.assertFalse(channel_pair.is_connected())

            proxy_mock.return_value.channel_grouping = HDAWGChannelGrouping.CHAN_GROUP_4x2

            channel_pair.connect_group(mock_device)
            self.assertTrue(channel_pair.is_connected())
            proxy_mock.assert_called_once_with(mock_device)
            self.assertIs(channel_pair.master_device, proxy_mock.return_value)
            self.assertIs(channel_pair.awg_module, channel_pair.master_device.api_session.awgModule.return_value)

    def test_set_volatile_parameters(self):
        mock_device = mock.Mock()

        parameters = {'a': 9}
        requested_changes = OrderedDict([(UserRegister.from_seqc(4), 2), (UserRegister.from_seqc(3), 6)])

        expected_user_reg_calls = [mock.call(*args) for args in requested_changes.items()]

        channel_pair = SingleDeviceChannelGroup(1, 2, 'foo', 3.4)

        channel_pair._current_program = 'active_program'
        with mock.patch.object(channel_pair._program_manager, 'get_register_values_to_update_volatile_parameters',
                               return_value=requested_changes) as get_reg_val:
            with mock.patch.object(channel_pair, 'user_register') as user_register:
                channel_pair.set_volatile_parameters('other_program', parameters)

                user_register.assert_not_called()
                get_reg_val.assert_called_once_with('other_program', parameters)

        with mock.patch.object(channel_pair._program_manager, 'get_register_values_to_update_volatile_parameters',
                               return_value=requested_changes) as get_reg_val:
            with mock.patch.object(channel_pair, 'user_register') as user_register:
                channel_pair.set_volatile_parameters('active_program', parameters)

                self.assertEqual(expected_user_reg_calls, user_register.call_args_list)
                get_reg_val.assert_called_once_with('active_program', parameters)

    def test_upload(self):
        mock_loop = mock.MagicMock(wraps=Loop(repetition_count=2,
                                              waveform=DummyWaveform(duration=192,
                                                                     sample_output=np.arange(192) / 192)))

        voltage_trafos = (lambda x: x, lambda x: x)

        with mock.patch('weakref.proxy'),\
             mock.patch('qupulse.hardware.awgs.zihdawg.make_compatible') as mock_make_compatible:
            channel_pair = SingleDeviceChannelGroup(1, 2, 'foo', 3.4)

            with self.assertRaisesRegex(HDAWGValueError, 'Channel ID'):
                channel_pair.upload('bar', mock_loop, ('A'), (None, 'A', None, None), voltage_trafos)
            with self.assertRaisesRegex(HDAWGValueError, 'Markers'):
                channel_pair.upload('bar', mock_loop, ('A', None), (None, 'A', None), voltage_trafos)
            with self.assertRaisesRegex(HDAWGValueError, 'transformations'):
                channel_pair.upload('bar', mock_loop, ('A', None), (None, 'A', None, None), voltage_trafos[:1])

            # TODO: draw the rest of the owl


@mock.patch('qupulse.hardware.awgs.zihdawg.ELFManager.AWGModule.compiler_upload', new_callable=mock.PropertyMock)
class ELFManagerTests(unittest.TestCase):
    def test_init(self, compiler_upload):
        manager = ELFManager(None)
        compiler_upload.assert_called_once_with(True)
        self.assertIsNone(manager._compile_job)
        self.assertIsNone(manager._upload_job)

    @unittest.skip("Write test after more hardware tests")
    def test_upload(self, compiler_upload):
        raise NotImplementedError()

    @unittest.skip("Write test after more hardware tests")
    def test_update_compile_job_status(self, compiler_upload):
        raise NotImplementedError()

    @unittest.skip("Write test after more hardware tests")
    def test_compile(self, compiler_upload):
        raise NotImplementedError()
