import unittest
import os
import sys

srcPath = os.path.dirname(os.path.abspath(__file__)).rsplit('tests',1)[0] + 'src'
sys.path.insert(0,srcPath)

from pulses.Serializer import Serializer
from pulses.SequencePulseTemplate import SequencePulseTemplate
from pulses.TablePulseTemplate import TablePulseTemplate
import json

class SerializerTest(unittest.TestCase):

    def test_serialization(self) -> None:
        table_foo = TablePulseTemplate(identifier='foo')
        table = TablePulseTemplate(measurement=False)
        sequence = SequencePulseTemplate([(table_foo, {}), (table, {})], [], identifier=None)
        serializer = Serializer()
        serialized = serializer.serialize(sequence)
        print(json.dumps(serialized, indent=4))
