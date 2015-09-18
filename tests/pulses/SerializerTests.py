import unittest
import sys
import os
import os.path
import json
from tempfile import TemporaryDirectory
from typing import Optional

srcPath = os.path.dirname(os.path.abspath(__file__)).rsplit('tests',1)[0] + 'src'
sys.path.insert(0,srcPath)

from pulses.Serializer import FilesystemBackend, Serializer, CachingBackend, Serializable
from pulses.TablePulseTemplate import TablePulseTemplate
from pulses.SequencePulseTemplate import SequencePulseTemplate
from pulses.Parameter import ParameterDeclaration
from tests.pulses.SerializationDummies import DummyStorageBackend


class DummySerializable(Serializable):

    def __init__(self, identifier: Optional[str]=None) -> None:
        super().__init__(identifier)

    @staticmethod
    def deserialize(serializer: Serializer, **kwargs) -> None:
        raise NotImplemented()

    def get_serialization_data(self, serializer: Serializer) -> None:
        raise NotImplemented()


class SerializableTests(unittest.TestCase):

    def test_identifier(self) -> None:
        serializable = DummySerializable()
        self.assertEqual(None, serializable.identifier)
        for identifier in [None, 'adsfi']:
            self.assertEqual(identifier, DummySerializable(identifier=identifier).identifier)
        with self.assertRaises(ValueError):
            DummySerializable('')


class FileSystemBackendTest(unittest.TestCase):

    def setUp(self) -> None:
        self.tmpdir = TemporaryDirectory()
        self.cwd = os.getcwd()
        os.chdir(self.tmpdir.name)
        dirname = 'fsbackendtest'
        os.mkdir(dirname) # replace by temporary directory
        self.backend = FilesystemBackend(dirname)
        self.testdata = 'dshiuasduzchjbfdnbewhsdcuzd'
        self.alternative_testdata = "8u993zhhbn\nb3tadgadg"
        self.identifier = 'some name'

    def tearDown(self) -> None:
        os.chdir(self.cwd)
        self.tmpdir.cleanup()

    def test_put_and_get_normal(self) -> None:
        # first put the data
        self.backend.put(self.identifier, self.testdata)

        # then retrieve it again
        data = self.backend.get(self.identifier)
        self.assertEqual(data, self.testdata)

    def test_put_file_exists_no_overwrite(self) -> None:
        name = 'test_put_file_exists_no_overwrite'
        self.backend.put(name, self.testdata)
        with self.assertRaises(FileExistsError):
            self.backend.put(name, self.alternative_testdata)
        self.assertEqual(self.testdata, self.backend.get(name))

    def test_put_file_exists_overwrite(self) -> None:
        name = 'test_put_file_exists_overwrite'
        self.backend.put(name, self.testdata)
        self.backend.put(name, self.alternative_testdata, overwrite=True)
        self.assertEqual(self.alternative_testdata, self.backend.get(name))

    def test_instantiation_fail(self) -> None:
        with self.assertRaises(NotADirectoryError):
            FilesystemBackend("C\\#~~")

    def test_exists(self) -> None:
        name = 'test_exists'
        self.backend.put(name, self.testdata)
        self.assertTrue(self.backend.exists(name))
        self.assertFalse(self.backend.exists('exists_not'))

    def test_get_not_existing(self) -> None:
        name = 'test_get_not_existing'
        with self.assertRaises(FileNotFoundError):
            self.backend.get(name)


class CachingBackendTests(unittest.TestCase):

    def setUp(self) -> None:
        self.dummy_backend = DummyStorageBackend()
        self.caching_backend = CachingBackend(self.dummy_backend)
        self.identifier = 'foo'
        self.testdata = 'foodata'
        self.alternative_testdata = 'atadoof'

    def test_put_and_get_normal(self) -> None:
        # first put the data
        self.caching_backend.put(self.identifier, self.testdata)

        # then retrieve it again
        data = self.caching_backend.get(self.identifier)
        self.assertEqual(data, self.testdata)

        data = self.caching_backend.get(self.identifier)
        self.assertEqual(data, self.testdata)
        self.assertEqual(1, self.dummy_backend.times_put_called)
        self.assertEqual(0, self.dummy_backend.times_get_called)

    def test_put_not_cached_existing_no_overwrite(self) -> None:
        self.dummy_backend.stored_items[self.identifier] = self.testdata
        with self.assertRaises(FileExistsError):
            self.caching_backend.put(self.identifier, self.alternative_testdata)

        self.caching_backend.get(self.identifier)
        data = self.caching_backend.get(self.identifier)
        self.assertEqual(self.testdata, data)
        self.assertEqual(1, self.dummy_backend.times_get_called)

    def test_put_not_cached_existing_overwrite(self) -> None:
        self.dummy_backend.stored_items[self.identifier] = self.testdata
        self.caching_backend.put(self.identifier, self.alternative_testdata, overwrite=True)

        data = self.caching_backend.get(self.identifier)
        self.assertEqual(self.alternative_testdata, data)
        self.assertEqual(0, self.dummy_backend.times_get_called)

    def test_put_cached_existing_no_overwrite(self) -> None:
        self.caching_backend.put(self.identifier, self.testdata)
        with self.assertRaises(FileExistsError):
            self.caching_backend.put(self.identifier, self.alternative_testdata)

        self.caching_backend.get(self.identifier)
        data = self.caching_backend.get(self.identifier)
        self.assertEqual(self.testdata, data)
        self.assertEqual(0, self.dummy_backend.times_get_called)

    def test_put_cached_existing_overwrite(self) -> None:
        self.caching_backend.put(self.identifier, self.testdata)
        self.caching_backend.put(self.identifier, self.alternative_testdata, overwrite=True)

        data = self.caching_backend.get(self.identifier)
        self.assertEqual(self.alternative_testdata, data)
        self.assertEqual(0, self.dummy_backend.times_get_called)

    def test_exists_cached(self) -> None:
        name = 'test_exists_cached'
        self.caching_backend.put(name, self.testdata)
        self.assertTrue(self.caching_backend.exists(name))

    def test_exists_not_cached(self) -> None:
        name = 'test_exists_not_cached'
        self.dummy_backend.put(name, self.testdata)
        self.assertTrue(self.caching_backend.exists(name))

    def test_exists_not(self) -> None:
        self.assertFalse(self.caching_backend.exists('test_exists_not'))

    def test_get_not_existing(self) -> None:
        name = 'test_get_not_existing'
        with self.assertRaises(FileNotFoundError):
            self.caching_backend.get(name)


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
