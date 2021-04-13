from unittest import TestCase, mock

from qupulse.hardware.feature_awg.features import ProgramManagement, VolatileParameters

from tests.hardware.feature_awg.awg_new_driver_base_tests import TestAWGChannelTuple, TestAWGDevice, TestAWGChannel, TestVolatileParameters


class ChannelTupleAdapterTest(TestCase):
    def setUp(self):
        self.device = TestAWGDevice("device")
        self.channels = [TestAWGChannel(0, self.device), TestAWGChannel(1, self.device)]
        self.tuple = TestAWGChannelTuple(0, device=self.device, channels=self.channels)

    def test_simple_properties(self):
        adapter = self.tuple.channel_tuple_adapter
        self.assertEqual(adapter.num_channels, len(self.channels))
        self.assertEqual(adapter.num_markers, 0)
        self.assertEqual(adapter.identifier, self.tuple.name)
        self.assertEqual(adapter.sample_rate, self.tuple.sample_rate)

    def test_upload(self):
        adapter = self.tuple.channel_tuple_adapter

        upload_kwargs = dict(name="upload_test",
                             program=mock.Mock(),
                             channels=('A', None),
                             voltage_transformation=(lambda x:x, lambda x:x**2),
                             markers=(), force=True)

        expected_kwargs = {**upload_kwargs, 'repetition_mode': None}
        expected_kwargs['marker_channels'] = expected_kwargs.pop('markers')

        with mock.patch.object(self.tuple[ProgramManagement], 'upload') as upload_mock:
            adapter.upload(**upload_kwargs)
            upload_mock.assert_called_once_with(**expected_kwargs)

    def test_arm(self):
        adapter = self.tuple.channel_tuple_adapter
        with mock.patch.object(self.tuple[ProgramManagement], 'arm') as arm_mock:
            adapter.arm('test_prog')
            arm_mock.assert_called_once_with('test_prog')

    def test_remove(self):
        adapter = self.tuple.channel_tuple_adapter
        with mock.patch.object(self.tuple[ProgramManagement], 'remove') as remove_mock:
            adapter.remove('test_prog')
            remove_mock.assert_called_once_with('test_prog')

    def test_clear(self):
        adapter = self.tuple.channel_tuple_adapter
        with mock.patch.object(self.tuple[ProgramManagement], 'clear') as clear_mock:
            adapter.clear()
            clear_mock.assert_called_once_with()

    def test_programs(self):
        adapter = self.tuple.channel_tuple_adapter
        with mock.patch.object(type(self.tuple[ProgramManagement]), 'programs', new_callable=mock.PropertyMock):
            self.assertIs(self.tuple[ProgramManagement].programs, adapter.programs)

    def test_set_volatile_parameters(self):
        adapter = self.tuple.channel_tuple_adapter

        self.tuple.add_feature(TestVolatileParameters(self.tuple))

        with mock.patch.object(self.tuple[VolatileParameters], 'set_volatile_parameters') as set_volatile_parameters_mock:
            adapter.set_volatile_parameters('wurst', {'a': 5.})
            set_volatile_parameters_mock.assert_called_once_with('wurst', {'a': 5.})
