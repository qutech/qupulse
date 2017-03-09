import unittest
import os.path
import json
from tempfile import TemporaryDirectory
from typing import Optional, Dict, Any

from qctoolkit.serialization import FilesystemBackend, Serializer, CachingBackend, Serializable
from qctoolkit.pulses.table_pulse_template import TablePulseTemplate
from qctoolkit.pulses.sequence_pulse_template import SequencePulseTemplate
from qctoolkit.pulses.parameters import ParameterDeclaration

from tests.serialization_dummies import DummyStorageBackend


class DummySerializable(Serializable):

    def __init__(self, data: str='foo', identifier: Optional[str]=None) -> None:
        super().__init__(identifier)
        self.data = data

    @staticmethod
    def deserialize(serializer: Serializer, data: str, identifier: Optional[str]=None) -> None:
        return DummySerializable(data, identifier)

    def get_serialization_data(self, serializer: Serializer) -> Dict[str, Any]:
        return dict(data=self.data)


class NestedDummySerializable(Serializable):

    def __init__(self, data: Serializable, identifier: Optional[str]=None) -> None:
        super().__init__(identifier)
        self.data = data

    @staticmethod
    def deserialize(serializer: Serializer, **kwargs) -> None:
        raise NotImplemented()

    def get_serialization_data(self, serializer: Serializer) -> Dict[str, Any]:
        return dict(data=serializer.dictify(self.data))


class SerializableTests(unittest.TestCase):

    def test_identifier(self) -> None:
        serializable = DummySerializable()
        self.assertEqual(None, serializable.identifier)
        for identifier in [None, 'adsfi']:
            self.assertEqual(identifier, DummySerializable(identifier=identifier).identifier)
        with self.assertRaises(ValueError):
            DummySerializable(identifier='')


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

    def setUp(self) -> None:
        self.backend = DummyStorageBackend()
        self.serializer = Serializer(self.backend)
        self.deserialization_data = dict(data='THIS IS DARTAA!',
                                         type=self.serializer.get_type_identifier(DummySerializable()))

    def test_serialize_subpulse_no_identifier(self) -> None:
        serializable = DummySerializable(data='bar')
        serialized = self.serializer.dictify(serializable)
        expected = serializable.get_serialization_data(self.serializer)
        expected['type'] = self.serializer.get_type_identifier(serializable)
        self.assertEqual(expected, serialized)

    def test_serialize_subpulse_identifier(self) -> None:
        serializable = DummySerializable(identifier='bar')
        serialized = self.serializer.dictify(serializable)
        self.assertEqual(serializable.identifier, serialized)

    def test_serialize_subpulse_duplicate_identifier(self) -> None:
        serializable = DummySerializable(identifier='bar')
        self.serializer.dictify(serializable)
        self.serializer.dictify(serializable)
        serializable = DummySerializable(data='this is other data than before', identifier='bar')
        with self.assertRaises(Exception):
            self.serializer.dictify(serializable)

    def test_collection_dictionaries_no_identifier(self) -> None:
        serializable = DummySerializable(data='bar')
        dictified = self.serializer._Serializer__collect_dictionaries(serializable)
        expected = {'': serializable.get_serialization_data(self.serializer)}
        expected['']['type'] = self.serializer.get_type_identifier(serializable)
        self.assertEqual(expected, dictified)

    def test_collection_dictionaries_identifier(self) -> None:
        serializable = DummySerializable(data='bar', identifier='foo')
        dicified = self.serializer._Serializer__collect_dictionaries(serializable)
        expected = {serializable.identifier: serializable.get_serialization_data(self.serializer)}
        expected[serializable.identifier]['type'] = self.serializer.get_type_identifier(serializable)
        self.assertEqual(expected, dicified)

    def test_dicitify_no_identifier_one_nesting_no_identifier(self) -> None:
        inner_serializable = DummySerializable(data='bar')
        serializable = NestedDummySerializable(data=inner_serializable)
        dicitified = self.serializer._Serializer__collect_dictionaries(serializable)
        expected = {'': serializable.get_serialization_data(self.serializer)}
        expected['']['type'] = self.serializer.get_type_identifier(serializable)
        self.assertEqual(expected, dicitified)

    def test_collection_dictionaries_no_identifier_one_nesting_identifier(self) -> None:
        inner_serializable = DummySerializable(data='bar', identifier='foo')
        serializable = NestedDummySerializable(data=inner_serializable)
        dicitified = self.serializer._Serializer__collect_dictionaries(serializable)
        expected = {'': serializable.get_serialization_data(self.serializer),
                    inner_serializable.identifier: inner_serializable.get_serialization_data(self.serializer)}
        expected['']['type'] = self.serializer.get_type_identifier(serializable)
        expected[inner_serializable.identifier]['type'] = self.serializer.get_type_identifier(inner_serializable)
        self.assertEqual(expected, dicitified)

    def test_collection_dictionaries_identifier_one_nesting_no_identifier(self) -> None:
        inner_serializable = DummySerializable(data='bar')
        serializable = NestedDummySerializable(data=inner_serializable, identifier='outer_foo')
        dicitified = self.serializer._Serializer__collect_dictionaries(serializable)
        expected = {serializable.identifier: serializable.get_serialization_data(self.serializer)}
        expected[serializable.identifier]['type'] = self.serializer.get_type_identifier(serializable)
        self.assertEqual(expected, dicitified)

    def test_collection_dictionaries_identifier_one_nesting_identifier(self) -> None:
        inner_serializable = DummySerializable(data='bar', identifier='foo')
        serializable = NestedDummySerializable(data=inner_serializable, identifier='outer_foo')
        dicitified = self.serializer._Serializer__collect_dictionaries(serializable)
        expected = {inner_serializable.identifier: inner_serializable.get_serialization_data(self.serializer),
                    serializable.identifier: serializable.get_serialization_data(self.serializer)}
        expected[serializable.identifier]['type'] = self.serializer.get_type_identifier(serializable)
        expected[inner_serializable.identifier]['type'] = self.serializer.get_type_identifier(inner_serializable)
        self.assertEqual(expected, dicitified)

    def __serialization_test_helper(self, serializable: Serializable, expected: Dict[str, str]) -> None:
        self.serializer.serialize(serializable)
        expected = {k: json.dumps(v, indent=4, sort_keys=True) for k,v in expected.items()}
        self.assertEqual(expected, self.backend.stored_items)

    def test_serialize_no_identifier(self) -> None:
        serializable = DummySerializable(data='bar')
        expected = {'main': serializable.get_serialization_data(self.serializer)}
        expected['main']['type'] = self.serializer.get_type_identifier(serializable)
        self.__serialization_test_helper(serializable, expected)

    def test_serialize_identifier(self) -> None:
        serializable = DummySerializable(data='bar', identifier='foo')
        expected = {serializable.identifier: serializable.get_serialization_data(self.serializer)}
        expected[serializable.identifier]['type'] = self.serializer.get_type_identifier(serializable)
        self.__serialization_test_helper(serializable, expected)

    def test_serialize_no_identifier_one_nesting_no_identifier(self) -> None:
        inner_serializable = DummySerializable(data='bar')
        serializable = NestedDummySerializable(data=inner_serializable)
        expected = {'main': serializable.get_serialization_data(self.serializer)}
        expected['main']['type'] = self.serializer.get_type_identifier(serializable)
        self.__serialization_test_helper(serializable, expected)

    def test_serialize_no_identifier_one_nesting_identifier(self) -> None:
        inner_serializable = DummySerializable(data='bar', identifier='foo')
        serializable = NestedDummySerializable(data=inner_serializable)
        expected = {'main': serializable.get_serialization_data(self.serializer),
                    inner_serializable.identifier: inner_serializable.get_serialization_data(self.serializer)}
        expected['main']['type'] = self.serializer.get_type_identifier(serializable)
        expected[inner_serializable.identifier]['type'] = self.serializer.get_type_identifier(inner_serializable)
        self.__serialization_test_helper(serializable, expected)

    def test_serialize_identifier_one_nesting_no_identifier(self) -> None:
        inner_serializable = DummySerializable(data='bar')
        serializable = NestedDummySerializable(data=inner_serializable, identifier='outer_foo')
        expected = {serializable.identifier: serializable.get_serialization_data(self.serializer)}
        expected[serializable.identifier]['type'] = self.serializer.get_type_identifier(serializable)
        self.__serialization_test_helper(serializable, expected)

    def test_serialize_identifier_one_nesting_identifier(self) -> None:
        inner_serializable = DummySerializable(data='bar', identifier='foo')
        serializable = NestedDummySerializable(data=inner_serializable, identifier='outer_foo')
        expected = {serializable.identifier: serializable.get_serialization_data(self.serializer),
                    inner_serializable.identifier: inner_serializable.get_serialization_data(self.serializer)}
        expected[serializable.identifier]['type'] = self.serializer.get_type_identifier(serializable)
        expected[inner_serializable.identifier]['type'] = self.serializer.get_type_identifier(inner_serializable)
        self.__serialization_test_helper(serializable, expected)

    def test_deserialize_dict(self) -> None:
        deserialized = self.serializer.deserialize(self.deserialization_data)
        self.assertIsInstance(deserialized, DummySerializable)
        self.assertEqual(self.deserialization_data['data'], deserialized.data)

    def test_deserialize_identifier(self) -> None:
        jsonized_data = json.dumps(self.deserialization_data, indent=4, sort_keys=True)
        identifier = 'foo'
        self.backend.put(identifier, jsonized_data)

        deserialized = self.serializer.deserialize(identifier)
        self.assertIsInstance(deserialized, DummySerializable)
        self.assertEqual(self.deserialization_data['data'], deserialized.data)

    def test_serialization_and_deserialization_combined(self) -> None:
        table_foo = TablePulseTemplate(identifier='foo')
        table_foo.add_entry('hugo', 2)
        table_foo.add_entry(ParameterDeclaration('albert', max=9.1), 'voltage')
        table = TablePulseTemplate()
        foo_mappings = dict(hugo='ilse', albert='albert', voltage='voltage')
        sequence = SequencePulseTemplate([(table_foo, foo_mappings, {}), (table, {}, {})], ['ilse', 'albert', 'voltage'], identifier=None)

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
