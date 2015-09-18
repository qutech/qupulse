import unittest
import sys
import os
import os.path
import json
from tempfile import TemporaryDirectory

srcPath = os.path.dirname(os.path.abspath(__file__)).rsplit('tests',1)[0] + 'src'
sys.path.insert(0,srcPath)

from pulses.Serializer import FilesystemBackend, Serializer, StorageBackend
from pulses.TablePulseTemplate import TablePulseTemplate
from pulses.SequencePulseTemplate import SequencePulseTemplate
from pulses.Parameter import ParameterDeclaration


class FileSystemBackendTest(unittest.TestCase):

    def setUp(self):
        self.tmpdir = TemporaryDirectory()
        self.cwd = os.getcwd()
        os.chdir(self.tmpdir.name)
        dirname = 'fsbackendtest'
        os.mkdir(dirname) # replace by temporary directory
        self.backend = FilesystemBackend(dirname)
        self.testdata = 'dshiuasduzchjbfdnbewhsdcuzd'
        self.identifier = 'some name'

    def tearDown(self):
        os.chdir(self.cwd)
        self.tmpdir.cleanup()

    def test_fsbackend(self):
        # first put the data
        self.backend.put(self.testdata, self.identifier)

        # then retrieve it again
        data = self.backend.get(self.identifier)
        self.assertEqual(data, self.testdata)


class DummyStorageBackend(StorageBackend):

    def __init__(self) -> None:
        self.stored_items = dict()

    def get(self, identifier: str) -> str:
        return self.stored_items[identifier]

    def put(self, data: str, identifier: str) -> None:
        self.stored_items[identifier] = data

class SerializerTests(unittest.TestCase):

    def test_serialization(self) -> None:
        table_foo = TablePulseTemplate(identifier='foo')
        table_foo.add_entry('hugo', 2)
        table_foo.add_entry(ParameterDeclaration('albert', max=9.1), 'voltage')
        table = TablePulseTemplate(measurement=True)
        foo_mappings = dict(hugo='ilse', albert='albert', voltage='voltage')
        sequence = SequencePulseTemplate([(table_foo, foo_mappings), (table, {})], ['ilse', 'albert', 'voltage'], identifier=None)

        storage = DummyStorageBackend()
        serializer = Serializer(storage)
        serializer.serialize(sequence)

        serialized_foo = storage.stored_items['foo']
        serialized_sequence = storage.stored_items['main']

        deserialized_sequence = serializer.deserialize('main')
        storage.stored_items = dict()
        serializer.serialize(deserialized_sequence)

        self.assertEqual(serialized_foo, storage.stored_items['foo'])
        self.assertEqual(serialized_sequence, storage.stored_items['main'])


    #def test_deserialization(self) -> None:
    #    serializer = Serializer(FilesystemBackend())
    #    obj = serializer.deserialize('main')
    #    print("hoooray")

    # def test_TablePulseTemplate(self):
    #     tpt = TablePulseTemplate()
    #     tpt.add_entry(1,1)
    #     tpt.add_entry('time', 'voltage')
    #     serializer = Serializer()
    #     result = serializer.serialize(tpt)
    #     self.fail() # TODO: this test was not complete when merged
    #
    # def test_SequencePulseTemplate(self):
    #     seq = SequencePulseTemplate([], [], identifier='sequence')
    #     serializer = Serializer()
    #     result = serializer.serialize(seq)
    #     self.fail() # TODO: this test was not complete when merged


if __name__ == "__main__":
    unittest.main(verbosity=2)
