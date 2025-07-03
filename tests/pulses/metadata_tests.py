import unittest

from qupulse.pulses.pulse_template import PulseTemplate, TemplateMetadata, MetadataComparison


class MetadataTest(unittest.TestCase):
    def test_default(self):
        tm = TemplateMetadata()
        self.assertIsNone(tm.to_single_waveform)

    def test_overwrite(self):
        tm = TemplateMetadata(to_single_waveform='always')
        self.assertEqual("always", tm.to_single_waveform)

    def test_custom_fields(self):
        tm = TemplateMetadata(foo=42)
        self.assertEqual(42, tm.foo)

        tm.bar = 9
        self.assertEqual(9, tm.bar)

    def test_repr(self):
        tm = TemplateMetadata()
        self.assertEqual("TemplateMetadata()", repr(tm))

        tm = TemplateMetadata(to_single_waveform='always')
        self.assertEqual("TemplateMetadata(to_single_waveform='always')", repr(tm))

        tm = TemplateMetadata(foo=42)
        self.assertEqual("TemplateMetadata(foo=42)", repr(tm))

    def test_serialization(self):
        tm = TemplateMetadata()
        self.assertEqual({}, tm.get_serialization_data())
        # check double because this was a bug before due to a missing copy
        self.assertEqual({}, tm.get_serialization_data())

        tm = TemplateMetadata(to_single_waveform='always')
        self.assertEqual({'to_single_waveform': 'always'}, tm.get_serialization_data())
        self.assertEqual({'to_single_waveform': 'always'}, tm.get_serialization_data())

        tm = TemplateMetadata(foo=42)
        self.assertEqual({'foo': 42}, tm.get_serialization_data())
        self.assertEqual({'foo': 42}, tm.get_serialization_data())

    def test_bool(self):
        tm = TemplateMetadata()
        self.assertFalse(tm)

        tm = TemplateMetadata(to_single_waveform='always')
        self.assertTrue(tm)

        tm = TemplateMetadata(foo=42)
        self.assertTrue(tm)
