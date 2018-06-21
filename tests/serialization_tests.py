import unittest
import os
import json
import zipfile
import typing
import json

from unittest import mock
from abc import ABCMeta, abstractmethod

from tempfile import TemporaryDirectory
from typing import Optional, Any

from qctoolkit.serialization import FilesystemBackend, CachingBackend, Serializable, JSONSerializableEncoder,\
    ZipFileBackend, AnonymousSerializable, DictBackend, PulseStorage, JSONSerializableDecoder, Serializer
from qctoolkit.expressions import ExpressionScalar

from tests.serialization_dummies import DummyStorageBackend
from tests.pulses.sequencing_dummies import DummyPulseTemplate


class DummySerializable(Serializable):

    def __init__(self, identifier: Optional[str]=None, **kwargs) -> None:
        super().__init__(identifier)
        for name in kwargs:
            setattr(self, name, kwargs[name])

    def get_serialization_data(self, serializer: Optional[Serializer]=None):
        local_data = dict(**self.__dict__)
        del local_data['_Serializable__identifier']
        if not serializer: # deprecated version for compatability with old serialization routine tests
            data = super().get_serialization_data()
            data.update(**local_data)
            return data
        else:
            local_data['identifier'] = self.identifier

            return local_data

    def __eq__(self, other) -> bool:
        if not isinstance(other, DummySerializable): return False
        return self.__dict__ == other.__dict__


class SerializableTests(metaclass=ABCMeta):

    @property
    @abstractmethod
    def class_to_test(self):
        pass

    @abstractmethod
    def make_kwargs(self):
        pass

    @abstractmethod
    def assert_equal_instance(self, lhs, rhs):
        pass

    def make_instance(self, identifier=None):
        return self.class_to_test(identifier=identifier, **self.make_kwargs())

    def make_serialization_data(self, identifier=None):
        data = {Serializable.type_identifier_name: self.class_to_test.get_type_identifier(), **self.make_kwargs()}
        if identifier:
            data[Serializable.identifier_name] = identifier
        return data

    def test_identifier(self) -> None:
        for identifier in [None, 'adsfi']:
            self.assertIs(identifier, self.make_instance(identifier).identifier)
        with self.assertRaises(ValueError):
            self.make_instance(identifier='')

    def test_get_type_identifier(self):
        instance = self.make_instance()

        self.assertEqual(instance.get_type_identifier(), type(instance).__module__ + '.' + type(instance).__name__)

    def test_serialization(self):
        for identifier in [None, 'some']:
            serialization_data = self.make_instance(identifier=identifier).get_serialization_data()
            expected = self.make_serialization_data(identifier=identifier)

            self.assertEqual(serialization_data, expected)

    def test_deserialiation(self) -> None:
        for identifier in [None, 'some']:
            serialization_data = self.make_serialization_data(identifier=identifier)
            del serialization_data[Serializable.type_identifier_name]
            if identifier:
                serialization_data['identifier'] = serialization_data[Serializable.identifier_name]
                del serialization_data[Serializable.identifier_name]
            instance = self.class_to_test.deserialize(**serialization_data)
            expected = self.make_instance(identifier=identifier)

            self.assert_equal_instance(expected, instance)


    def test_serialization_and_deserialization(self):
        instance = self.make_instance('blub')
        backend = DummyStorageBackend()
        storage = PulseStorage(backend)

        storage['blub'] = instance

        storage.flush()
        storage.clear()

        other_instance = typing.cast(self.class_to_test, storage['blub'])
        self.assert_equal_instance(instance, other_instance)


class DummySerializableTests(SerializableTests, unittest.TestCase):
    @property
    def class_to_test(self):
        return DummySerializable

    def make_kwargs(self):
        return {'data': 'blubber', 'test_dict': {'foo': 'bar', 'no': 17.3}}

    def assert_equal_instance(self, lhs, rhs):
        self.assertEqual(lhs.identifier, rhs.identifier)
        self.assertEqual(lhs.data, rhs.data)


class DummyPulseTemplateSerializationtests(SerializableTests, unittest.TestCase):
    @property
    def class_to_test(self):
        return DummyPulseTemplate

    def make_kwargs(self):
        return {
            'requires_stop': True,
            'is_interruptable': True,
            'parameter_names': {'foo', 'bar'},
            'defined_channels': {'default', 'not_default'},
            'duration': ExpressionScalar('17.3*foo+bar'),
            'measurement_names': {'hugo'},
            'integrals': {'default': ExpressionScalar(19.231)}
        }

    def assert_equal_instance(self, lhs, rhs):
        self.assertEqual(lhs.compare_key, rhs.compare_key)
        self.assertEqual(lhs.identifier, rhs.identifier)


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
        with self.assertRaisesRegex(KeyError, name):
            self.backend.get(name)


class ZipFileBackendTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_init(self):
        with TemporaryDirectory() as tmp_dir:

            with self.assertRaises(NotADirectoryError):
                ZipFileBackend(os.path.join(tmp_dir, 'fantasie', 'mehr_phantasie'))

            root = os.path.join(tmp_dir, 'root.zip')

            ZipFileBackend(root)

            self.assertTrue(zipfile.is_zipfile(root))

    def test_init_keeps_data(self):
        with TemporaryDirectory() as tmp_dir:
            root = os.path.join(tmp_dir, 'root.zip')
            with zipfile.ZipFile(root, mode='w', compression=zipfile.ZIP_DEFLATED) as zip_file:
                zip_file.writestr('test_file.txt', 'chichichi')

            ZipFileBackend(root)

            with zipfile.ZipFile(root, 'r') as zip_file:
                ma_string = zip_file.read('test_file.txt')
                self.assertEqual(b'chichichi', ma_string)

    def test_path(self):
        with TemporaryDirectory() as tmp_dir:
            root = os.path.join(tmp_dir, 'root.zip')
            be = ZipFileBackend(root)
            self.assertEqual(be._path('foo'), 'foo.json')

    def test_exists(self):
        with TemporaryDirectory() as tmp_dir:
            root = os.path.join(tmp_dir, 'root.zip')
            be = ZipFileBackend(root)

            self.assertFalse(be.exists('foo'))

            with zipfile.ZipFile(root, mode='w', compression=zipfile.ZIP_DEFLATED) as zip_file:
                zip_file.writestr('foo.json', 'chichichi')

            self.assertTrue(be.exists('foo'))

    def test_put(self):
        with TemporaryDirectory() as tmp_dir:
            root = os.path.join(tmp_dir, 'root.zip')

            be = ZipFileBackend(root)

            be.put('foo', 'foo_data')

            with zipfile.ZipFile(root, 'r') as zip_file:
                ma_string = zip_file.read('foo.json')
                self.assertEqual(b'foo_data', ma_string)

            with self.assertRaises(FileExistsError):
                be.put('foo', 'bar_data')
            with zipfile.ZipFile(root, 'r') as zip_file:
                ma_string = zip_file.read('foo.json')
                self.assertEqual(b'foo_data', ma_string)

            be.put('foo', 'foo_bar_data', overwrite=True)
            with zipfile.ZipFile(root, 'r') as zip_file:
                ma_string = zip_file.read('foo.json')
                self.assertEqual(b'foo_bar_data', ma_string)

    def test_get(self):
        with TemporaryDirectory() as tmp_dir:
            root = os.path.join(tmp_dir, 'root.zip')
            be = ZipFileBackend(root)

            with self.assertRaises(KeyError):
                be.get('foo')

            data = 'foo_data'
            with zipfile.ZipFile(root, 'a') as zip_file:
                zip_file.writestr('foo.json', data)

            self.assertEqual(be.get('foo'), data)

    def test_update(self):
        with TemporaryDirectory() as tmp_dir:
            root = os.path.join(tmp_dir, 'root.zip')
            be = ZipFileBackend(root)

            be.put('foo', 'foo_data')
            be.put('bar', 'bar_data')

            be._update('foo.json', 'foo_bar_data')

            self.assertEqual(be.get('foo'), 'foo_bar_data')
            self.assertEqual(be.get('bar'), 'bar_data')


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
        with self.assertRaises(KeyError):
            self.caching_backend.get(name)


class DictBackendTests(unittest.TestCase):
    def setUp(self):
        self.backend =DictBackend()

    def test_put(self):
        self.backend.put('a', 'data')

        self.assertEqual(self.backend.storage, {'a': 'data'})

        with self.assertRaises(FileExistsError):
            self.backend.put('a', 'data2')

    def test_get(self):
        self.backend.put('a', 'data')
        self.backend.put('b', 'data2')

        self.assertEqual(self.backend.get('a'), 'data')
        self.assertEqual(self.backend.get('b'), 'data2')

    def test_exists(self):
        self.backend.put('a', 'data')
        self.backend.put('b', 'data2')

        self.assertTrue(self.backend.exists('a'))
        self.assertTrue(self.backend.exists('b'))
        self.assertFalse(self.backend.exists('c'))


class PulseStorageTests(unittest.TestCase):
    def setUp(self):
        self.backend = DummyStorageBackend()
        self.storage = PulseStorage(self.backend)

    def test_deserialize(self):
        obj = {'my_obj': 'tröt', 'wurst': [12, 3, 4]}
        serialized = json.dumps(obj)
        deserialized = self.storage._deserialize(serialized)
        self.assertEqual(deserialized, obj)

    def test_contains(self):
        instance = DummySerializable(identifier='my_id')

        self.assertNotIn(instance.identifier, self.storage)
        self.backend[instance.identifier] = 'dummy_string'

        self.assertIn(instance.identifier, self.storage)

        del self.backend[instance.identifier]
        self.assertNotIn(instance.identifier, self.storage)

        self.storage[instance.identifier] = instance
        self.assertIn(instance.identifier, self.storage)

    def test_getitem(self):
        instance = DummySerializable(identifier='my_id')

        self.storage.temporary_storage['asd'] = PulseStorage.StorageEntry(serialization='foobar', serializable=instance)

        self.assertIs(self.storage['asd'], instance)
        with self.assertRaises(KeyError):
            _ = self.storage['asdf']

        obj = {'jkl': [1, 2, 3, 4], 'wupper': 'tal'}
        self.backend['asdf'] = json.dumps(obj)

        other_obj = self.storage['asdf']
        self.assertEqual(obj, other_obj)

        self.assertIn('asdf', self.storage.temporary_storage)

    def test_setitem(self):
        instance_1 = DummySerializable(identifier='my_id_1')
        instance_2 = DummySerializable(identifier='my_id_2')

        def overwrite(identifier, serializable):
            self.assertFalse(overwrite.called)
            self.assertEqual(identifier, 'my_id')
            self.assertIs(serializable, instance_1)
            overwrite.wrapped(identifier, serializable)
            overwrite.called = True
        overwrite.called = False
        overwrite.wrapped = self.storage.overwrite

        setattr(self.storage, 'overwrite', overwrite)

        self.storage['my_id'] = instance_1

        self.storage['my_id'] = instance_1

        with self.assertRaisesRegex(RuntimeError, 'assigned twice'):
            self.storage['my_id'] = instance_2

    def test_overwrite(self):

        encode_mock = mock.Mock(return_value='asd')

        instance = DummySerializable(identifier='my_id')

        with mock.patch.object(JSONSerializableEncoder, 'encode', new=encode_mock):
            self.storage.overwrite('my_id', instance)

        self.assertEqual(encode_mock.call_count, 1)
        self.assertEqual(encode_mock.call_args, mock.call(instance.get_serialization_data()))

        self.assertEqual(self.storage._temporary_storage, {'my_id': self.storage.StorageEntry('asd', instance)})

    def test_flush(self):
        instance_1 = DummySerializable(identifier='my_id_1')
        instance_2 = DummySerializable(identifier='my_id_2')

        def get_expected():
            return {identifier: serialized
                    for identifier, (serialized, _) in self.storage.temporary_storage.items()}

        self.storage['my_id_1'] = instance_1
        self.storage['my_id_2'] = instance_2

        self.assertFalse(self.backend.stored_items)

        self.storage.flush()

        self.assertEqual(get_expected(), self.backend.stored_items)

    def test_flush_with_ignore(self):
        instance_1 = DummySerializable(identifier='my_id_1')
        instance_2 = DummySerializable(identifier='my_id_2')
        instance_3 = DummySerializable(identifier='my_id_3')

        ignore = ['my_id_1', 'my_id_3']

        def get_expected():
            return {identifier: serialized
                    for identifier, (serialized, _) in self.storage.temporary_storage.items()
                    if identifier not in ignore}

        self.storage['my_id_1'] = instance_1
        self.storage['my_id_2'] = instance_2
        self.storage['my_id_3'] = instance_3

        self.assertFalse(self.backend.stored_items)

        self.storage.flush(to_ignore=ignore)

        self.assertEqual(get_expected(), self.backend.stored_items)

    def test_clear(self):
        instance_1 = DummySerializable(identifier='my_id_1')
        instance_2 = DummySerializable(identifier='my_id_2')
        instance_3 = DummySerializable(identifier='my_id_3')

        self.storage['my_id_1'] = instance_1
        self.storage['my_id_2'] = instance_2
        self.storage['my_id_3'] = instance_3

        self.storage.clear()

        self.assertFalse(self.storage.temporary_storage)

    def test_flush_on_destroy_object(self) -> None:
        instance_1 = DummySerializable(identifier='my_id_1')
        backend = DummyStorageBackend()

        storage = PulseStorage(backend)
        storage['my_id_1'] = instance_1
        self.assertNotIn('my_id_1', backend.stored_items)
        del storage

        self.assertIn('my_id_1', backend.stored_items)

    def test_beautified_json(self) -> None:
        data = {'e': 89, 'b': 151, 'c': 123515, 'a': 123, 'h': 2415}
        template = DummySerializable(data=data)
        pulse_storage = PulseStorage(DummyStorageBackend())
        pulse_storage['foo'] = template
        pulse_storage.flush()
        expected = """{
    \"#type\": \"""" + DummySerializable.get_type_identifier() + """\",
    \"data\": {
        \"a\": 123,
        \"b\": 151,
        \"c\": 123515,
        \"e\": 89,
        \"h\": 2415
    }
}"""
        self.assertEqual(expected, pulse_storage._storage_backend['foo'])


class JSONSerializableDecoderTests(unittest.TestCase):
    def test_filter_serializables(self):
        storage = dict(asd='asd_value')

        decoder = JSONSerializableDecoder(storage)

        no_type_dict = dict(bla=9)
        self.assertIs(decoder.filter_serializables(no_type_dict), no_type_dict)

        reference_dict = {'#type': 'reference'}
        with self.assertRaisesRegex(RuntimeError, 'identifier'):
            decoder.filter_serializables(reference_dict)

        reference_dict = {'#type': 'reference', '#identifier': 'asd'}
        self.assertIs(storage['asd'], decoder.filter_serializables(reference_dict))

        dummy_dict = DummySerializable(data='foo_bar').get_serialization_data()
        decoded = decoder.filter_serializables(dummy_dict)
        self.assertIsInstance(decoded, DummySerializable)
        self.assertEqual(decoded.data, 'foo_bar')

    def test_decode(self):
        encoded = r'{"#type": "%s",' \
                  r'"#identifier": "my_id",' \
                  r'"data": {"#type": "reference",' \
                           r'"#identifier": "referenced"}}' % DummySerializable.get_type_identifier()

        referenced = [1, 2, 3]

        storage = dict(referenced=referenced)

        decoder = JSONSerializableDecoder(storage)

        decoded = decoder.decode(encoded)

        self.assertIsInstance(decoded, DummySerializable)
        self.assertIs(decoded.data, referenced)


class JSONSerializableEncoderTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_default(self):
        existing_serializable = DummySerializable(identifier='existing_id')
        storage = {'existing_id': existing_serializable}

        encoder = JSONSerializableEncoder(storage)

        test_set = {1, 2, 3, 4}
        self.assertEqual(test_set, set(encoder.default(test_set)))
        self.assertEqual(1, len(storage))

        class A(AnonymousSerializable):
            anonymous_serialization_data = [1, 2, 3]

            def get_serialization_data(self):
                return self.anonymous_serialization_data

        self.assertIs(encoder.default(A()), A.anonymous_serialization_data)

        expected_conversion = {'#type': 'reference',
                               '#identifier': 'existing_id'}
        self.assertEqual(encoder.default(existing_serializable), expected_conversion)

        new_serializable = DummySerializable(identifier='new_id', data=[1, 2, 3])
        expected_conversion = {'#type': 'reference',
                               '#identifier': 'new_id'}
        encoded = encoder.default(new_serializable)
        self.assertEqual(expected_conversion, encoded)

        self.assertIn('new_id', storage)
        self.assertIs(storage['new_id'], new_serializable)

        no_id_serializable = DummySerializable()
        encoded = encoder.default(no_id_serializable)
        self.assertEqual(no_id_serializable.get_serialization_data(), encoded)

        self.assertEqual(set(storage.keys()), {'existing_id', 'new_id'})
        self.assertIs(storage['new_id'], new_serializable)
        self.assertIs(storage['existing_id'], existing_serializable)

    def test_encoding(self):
        class A(AnonymousSerializable):
            anonymous_serialization_data = [1, 2, 3]

            def get_serialization_data(self):
                return self.anonymous_serialization_data

        inner_anon = DummySerializable(data=[A(), 1])
        inner_named = DummySerializable(data='bar', identifier='inner')
        inner_known = DummySerializable(data='bar', identifier='known')

        outer = DummySerializable(data=[inner_named, inner_anon, inner_known])

        inner_known_storage = [567]
        storage = dict(known=inner_known_storage)
        encoder = JSONSerializableEncoder(storage)

        encoded = encoder.encode(outer)

        expected = {"#type": DummySerializable.get_type_identifier(),
                    "data": [
                        {'#type': 'reference',
                         '#identifier': 'inner'},
                        {'#type': DummySerializable.get_type_identifier(),
                         'data': [[1, 2, 3], 1]},
                        {'#type': 'reference',
                         '#identifier': 'known'}
                    ]
                    }
        expected_encoded = json.dumps(expected)
        self.assertEqual(expected_encoded, encoded)

        self.assertEqual(set(storage.keys()), {'inner', 'known'})
        self.assertIs(storage['inner'], inner_named)
        self.assertIs(storage['known'], inner_known_storage)



########################################################################################################################
################################ tests for old architecture, now deprecated ############################################
########################################################################################################################

import warnings
from typing import Dict
from qctoolkit.serialization import ExtendedJSONEncoder, Serializer
from qctoolkit.pulses.table_pulse_template import TablePulseTemplate
from qctoolkit.pulses.sequence_pulse_template import SequencePulseTemplate


class NestedDummySerializable(Serializable):

    def __init__(self, data: Serializable, identifier: Optional[str]=None) -> None:
        super().__init__(identifier)
        self.data = data

    @classmethod
    def deserialize(cls, serializer: Optional[Serializer]=None, **kwargs) -> None:
        if serializer:
            data = serializer.deserialize(kwargs['data'])
        else:
            data = kwargs['data']
        return NestedDummySerializable(data, identifier=kwargs['identifier'])

    def get_serialization_data(self, serializer: Optional[Serializer]=None) -> Dict[str, Any]:
        if not serializer:
            data = super().get_serialization_data()
            data['data'] = self.data
        else:
            data = dict()
            data['data'] = serializer.dictify(self.data)
        return data

    def __eq__(self, other) -> None:
        return self.data, self.identifier == other.data, other.identifier


class SerializerTests(unittest.TestCase):

    def setUp(self) -> None:
        self.warn_collection = warnings.catch_warnings(record=True)
        warnings.simplefilter("ignore", category=DeprecationWarning)

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
        table_foo = TablePulseTemplate(identifier='foo', entries={'default': [('hugo', 2),
                                                                              ('albert', 'voltage')]},
                                       parameter_constraints=['albert<9.1'])
        table = TablePulseTemplate({'default': [('t', 0)]})

        foo_mappings = dict(hugo='ilse', albert='albert', voltage='voltage')
        sequence = SequencePulseTemplate((table_foo, foo_mappings, dict()),
                                         (table, dict(t=0), dict()),
                                         identifier=None)
        self.assertEqual({'ilse', 'albert', 'voltage'}, sequence.parameter_names)

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


class TriviallyRepresentableEncoderTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_encoding(self):
        class A(AnonymousSerializable):
            def get_serialization_data(self):
                return 'aaa'

        class B:
            pass

        encoder = ExtendedJSONEncoder()

        a = A()
        self.assertEqual(encoder.default(a), 'aaa')

        with self.assertRaises(TypeError):
            encoder.default(B())

        self.assertEqual(encoder.default({'a', 1}), list({'a', 1}))
