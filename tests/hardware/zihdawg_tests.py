import unittest
from unittest import mock
from collections import OrderedDict

import numpy as np

from qupulse.utils.types import TimeType
from qupulse._program._loop import Loop
from tests.pulses.sequencing_dummies import DummyWaveform
from qupulse.hardware.awgs.zihdawg import HDAWGChannelPair, HDAWGRepresentation, HDAWGValueError, UserRegister,\
    ConstantParameter


class HDAWGRepresentationTests(unittest.TestCase):
    def test_init(self):
        """We do not test anything lab one related"""
        device_serial = 'dev6ä6ä6'
        device_interface = 'telepathy'
        data_server_addr = 'asd'
        data_server_port = 42
        api_level_number = 23

        with \
                mock.patch('zhinst.utils.api_server_version_check') as mock_version_check,\
                mock.patch('zhinst.ziPython.ziDAQServer') as mock_daq_server, \
                mock.patch('qupulse.hardware.awgs.zihdawg.HDAWGRepresentation._initialize') as mock_init, \
                mock.patch('qupulse.hardware.awgs.zihdawg.HDAWGChannelPair') as mock_channel_pair,\
                mock.patch('zhinst.utils.disable_everything') as mock_reset:

            representation = HDAWGRepresentation(device_serial,
                                                 device_interface,
                                                 data_server_addr, data_server_port, api_level_number, False, 1.3)

            self.assertIs(representation.api_session, mock_daq_server.return_value)
            mock_daq_server.assert_called_once_with(data_server_addr, data_server_port, api_level_number)

            mock_version_check.assert_called_once_with(representation.api_session)
            representation.api_session.connectDevice.assert_called_once_with(device_serial, device_interface)
            self.assertEqual(device_serial, representation.serial)

            mock_reset.assert_not_called()
            mock_init.assert_called_once_with()

            pair_calls = [mock.call(representation, (2*i+1, 2*i+2), str(device_serial) + post_fix, 1.3)
                          for i, post_fix in enumerate(['_AB', '_CD', '_EF', '_GH'])]
            for c1, c2 in zip(pair_calls, mock_channel_pair.call_args_list):
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

            pair_calls = [mock.call(representation, (2*i+1, 2*i+2), str(device_serial) + post_fix, 20)
                          for i, post_fix in enumerate(['_AB', '_CD', '_EF', '_GH'])]
            self.assertEqual(pair_calls, mock_channel_pair.call_args_list)

            self.assertIs(representation.channel_pair_AB, mock_channel_pair.return_value)
            self.assertIs(representation.channel_pair_CD, mock_channel_pair.return_value)
            self.assertIs(representation.channel_pair_EF, mock_channel_pair.return_value)
            self.assertIs(representation.channel_pair_GH, mock_channel_pair.return_value)


class HDAWGChannelPairTests(unittest.TestCase):
    def test_init(self):
        with mock.patch('weakref.proxy') as proxy_mock:
            mock_device = mock.Mock()

            channel_pair = HDAWGChannelPair(mock_device, (3, 4), 'foo', 3.4)

            self.assertEqual(channel_pair.timeout, 3.4)
            self.assertEqual(channel_pair._channels, (3, 4))
            self.assertEqual(channel_pair.awg_group_index, 1)

            proxy_mock.assert_called_once_with(mock_device)
            self.assertIs(channel_pair.device, proxy_mock.return_value)

            self.assertIs(channel_pair.awg_module, channel_pair.device.api_session.awgModule.return_value)
            self.assertEqual(channel_pair.num_channels, 2)
            self.assertEqual(channel_pair.num_markers, 4)

    def test_set_volatile_parameters(self):
        mock_device = mock.Mock()

        parameters = {'a': ConstantParameter(9)}
        requested_changes = OrderedDict([(UserRegister.from_seqc(4), 2), (UserRegister.from_seqc(3), 6)])

        expected_user_reg_calls = [mock.call(*args) for args in requested_changes.items()]

        channel_pair = HDAWGChannelPair(mock_device, (3, 4), 'foo', 3.4)

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
            channel_pair = HDAWGChannelPair(mock.Mock(), (3, 4), 'foo', 3.4)

            with self.assertRaisesRegex(HDAWGValueError, 'Channel ID'):
                channel_pair.upload('bar', mock_loop, ('A'), (None, 'A', None, None), voltage_trafos)
            with self.assertRaisesRegex(HDAWGValueError, 'Markers'):
                channel_pair.upload('bar', mock_loop, ('A', None), (None, 'A', None), voltage_trafos)
            with self.assertRaisesRegex(HDAWGValueError, 'transformations'):
                channel_pair.upload('bar', mock_loop, ('A', None), (None, 'A', None, None), voltage_trafos[:1])

            # TODO: draw the rest of the owl
