import unittest
import os
import sys
import zipfile
import typing
import json

from unittest import mock
from abc import ABCMeta, abstractmethod

from tempfile import TemporaryDirectory, NamedTemporaryFile, TemporaryFile
from typing import Optional, Any, Tuple

from qupulse.serialization import FilesystemBackend, CachingBackend, Serializable, JSONSerializableEncoder,\
    ZipFileBackend, AnonymousSerializable, DictBackend, PulseStorage, JSONSerializableDecoder, Serializer,\
    get_default_pulse_registry, set_default_pulse_registry, new_default_pulse_registry, SerializableMeta, \
    PulseRegistryType, DeserializationCallbackFinder, StorageBackend

from qupulse.expressions import ExpressionScalar

from tests.serialization_dummies import DummyStorageBackend
from tests.pulses.sequencing_dummies import DummyPulseTemplate


class DummySerializable(Serializable):

    def __init__(self, identifier: Optional[str]=None, registry: PulseRegistryType=None, **kwargs) -> None:
        super().__init__(identifier)
        for name in kwargs:
            setattr(self, name, kwargs[name])

        self._register(registry=registry)

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
    def assertEqual(self, first, second, msg=None):
        # We use the id based hashing and comparison in other places. For easy testing, we patch the __eq__ here
        # temporarily.
        def dummy_pulse_template_equal(lhs, rhs):
            return lhs.compare_key == rhs.compare_key

        with mock.patch.object(DummyPulseTemplate, '__eq__', dummy_pulse_template_equal):
            unittest.TestCase.assertEqual(self, first, second, msg=msg)

    @property
    @abstractmethod
    def class_to_test(self) -> typing.Any:
        pass

    @abstractmethod
    def make_kwargs(self) -> dict:
        pass

    def assert_equal_instance(self, lhs, rhs):
        self.assert_equal_instance_except_id(lhs, rhs)
        self.assertEqual(lhs.identifier, rhs.identifier)

    @abstractmethod
    def assert_equal_instance_except_id(self, lhs, rhs):
        pass

    def make_instance(self, identifier=None, registry=None):
        return self.class_to_test(identifier=identifier, registry=registry, **self.make_kwargs())

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
            serialization_data = self.make_instance(identifier=identifier, registry=None).get_serialization_data()
            expected = self.make_serialization_data(identifier=identifier)

            self.assertEqual(serialization_data, expected)

    def test_deserialization(self) -> None:
        registry_1 = dict()
        registry_2 = dict()
        for identifier in [None, 'some']:
            serialization_data = self.make_serialization_data(identifier=identifier)
            del serialization_data[Serializable.type_identifier_name]
            if identifier:
                serialization_data['identifier'] = serialization_data[Serializable.identifier_name]
                del serialization_data[Serializable.identifier_name]
            instance = self.class_to_test.deserialize(**serialization_data, registry=registry_1)

            if identifier:
                self.assertIs(registry_1[identifier], instance)
            expected = self.make_instance(identifier=identifier, registry=registry_2)

            self.assert_equal_instance(expected, instance)

    def test_serialization_and_deserialization(self):
        registry = dict()

        instance = self.make_instance('blub', registry=registry)
        backend = DummyStorageBackend()
        storage = PulseStorage(backend)

        storage['blub'] = instance

        storage.clear()
        set_default_pulse_registry(dict())

        other_instance = typing.cast(self.class_to_test, storage['blub'])
        self.assert_equal_instance(instance, other_instance)

        self.assertIs(registry['blub'], instance)
        self.assertIs(get_default_pulse_registry()['blub'], other_instance)
        set_default_pulse_registry(None)

    def test_duplication_error(self):
        registry = dict()

        inst = self.make_instance('blub', registry=registry)

        # ensure that no two objects with same id can be created
        with self.assertRaises(RuntimeError):
            self.make_instance('blub', registry=registry)

    def test_manual_garbage_collect(self):
        import weakref
        registry = weakref.WeakValueDictionary()

        inst = self.make_instance('blub', registry=registry)

        import gc
        gc_state = gc.isenabled()
        try:
            # Disable garbage collection and create circular references to check whether manual gc invocation works
            gc.disable()

            temp = ({}, {})
            temp[0][0] = temp[1]
            temp[1][0] = temp[0]
            temp[0][1] = inst

            del inst
            del temp
            with mock.patch('qupulse.serialization.gc.collect', mock.MagicMock(side_effect=gc.collect)) as mocked_gc:
                self.make_instance('blub', registry=registry)
                mocked_gc.assert_called_once_with(2)
        finally:
            # reenable gc if it was enabled before
            if gc_state:
                gc.enable()

    def test_no_registration_before_correct_serialization(self) -> None:
        class RegistryStub:
            def __init__(self) -> None:
                self.storage = dict()

            def __setitem__(self, key: str, value: Serializable) -> None:
                serialization_data = value.get_serialization_data()
                serialization_data.pop(Serializable.type_identifier_name)
                serialization_data.pop(Serializable.identifier_name)
                self.storage[key] = (value, serialization_data)

            def __getitem__(self, key: str) -> Tuple[Serializable, Dict[str, Any]]:
                return self.storage[key]

            def __contains__(self, key: str) -> bool:
                return key in self.storage

        registry = RegistryStub()
        identifier = 'foo'
        instance = self.make_instance(identifier=identifier, registry=registry)
        self.assertIs(instance, registry[identifier][0])
        stored_instance = self.class_to_test.deserialize(identifier=identifier, registry=dict(), **(registry[identifier][1]))
        self.assert_equal_instance(instance, stored_instance)

    def test_renamed(self) -> None:
        registry = dict()
        instance = self.make_instance('hugo', registry=registry)
        renamed_instance = instance.renamed('ilse', registry=registry)
        self.assertEqual(renamed_instance.identifier, 'ilse')
        self.assert_equal_instance_except_id(instance, renamed_instance)

    def test_renamed_of_anonymous(self):
        registry = dict()
        instance = self.make_instance(None, registry=registry)
        renamed_instance = instance.renamed('ilse', registry=registry)
        self.assertEqual(renamed_instance.identifier, 'ilse')
        self.assert_equal_instance_except_id(instance, renamed_instance)
        
    def test_conversion(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            source_backend = DummyStorageBackend()
            instance = self.make_instance(identifier='foo', registry=dict())
            serializer = Serializer(source_backend)
            serializer.serialize(instance)
            del serializer

            dest_backend = DummyStorageBackend()
            convert_pulses_in_storage(source_backend, dest_backend)
            pulse_storage = PulseStorage(dest_backend)
            converted = pulse_storage['foo']
            self.assert_equal_instance(instance, converted)


class DummySerializableTests(SerializableTests, unittest.TestCase):
    @property
    def class_to_test(self):
        return DummySerializable

    def make_kwargs(self):
        return {'data': 'blubber', 'test_dict': {'foo': 'bar', 'no': 17.3}}

    def assert_equal_instance_except_id(self, lhs, rhs):
        self.assertEqual(lhs.data, rhs.data)


class DummyPulseTemplateSerializationTests(SerializableTests, unittest.TestCase):
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

    def assert_equal_instance_except_id(self, lhs, rhs):
        self.assertEqual(lhs.compare_key, rhs.compare_key)


@mock.patch.multiple(StorageBackend, __abstractmethods__=set())
class StorageBackendTest(unittest.TestCase):
    """Testing common methods implemented in StorageBackend base class based on the abstract methods implemented
    by subclasses."""

    def test_setitem(self) -> None:
        with mock.patch.object(StorageBackend, 'put') as put_mock:
            storage = StorageBackend()
            storage["foo"] = "bar"
            self.assertEqual(mock.call('foo', 'bar'), put_mock.call_args)

        with mock.patch.object(StorageBackend, 'put', side_effect=FileExistsError()) as put_mock:
            storage = StorageBackend()
            with self.assertRaises(FileExistsError):
                storage["foo"] = "bar"
            self.assertEqual(mock.call('foo', 'bar'), put_mock.call_args)

    def test_getitem(self) -> None:
        expected = "bar"
        with mock.patch.object(StorageBackend, 'get', return_value=expected) as get_mock:
            storage = StorageBackend()
            foo = storage["foo"]
            self.assertEqual(expected, foo)
            self.assertEqual(mock.call("foo"), get_mock.call_args)

        with mock.patch.object(StorageBackend, 'get', side_effect=KeyError()) as get_mock:
            storage = StorageBackend()
            with self.assertRaises(KeyError):
                storage['foo']
            self.assertEqual(mock.call('foo'), get_mock.call_args)

    def test_contains(self) -> None:
        with mock.patch.object(StorageBackend, 'exists', return_value=True) as exists_mock:
            storage = StorageBackend()
            self.assertTrue('foo' in storage)
            self.assertEqual(mock.call('foo'), exists_mock.call_args)

        with mock.patch.object(StorageBackend, 'exists', return_value=False) as exists_mock:
            storage = StorageBackend()
            self.assertFalse('foo' in storage)
            self.assertEqual(mock.call('foo'), exists_mock.call_args)

    def test_delitem(self) -> None:
        with mock.patch.object(StorageBackend, 'delete') as delete_mock:
            storage = StorageBackend()
            del(storage['foo'])
            self.assertEqual(mock.call('foo'), delete_mock.call_args)

        with mock.patch.object(StorageBackend, 'delete', side_effect=KeyError()) as delete_mock:
            storage = StorageBackend()
            with self.assertRaises(KeyError):
                del(storage['foo'])
            self.assertEqual(mock.call('foo'), delete_mock.call_args)

    def test_list_contents(self) -> None:
        expected = {'hugo', 'ilse', 'foo.bar'}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            with mock.patch.object(StorageBackend, '__iter__', return_value=iter(expected)) as iter_mock:
                storage = StorageBackend()
                self.assertEqual(expected, storage.list_contents())
                self.assertEqual(1, iter_mock.call_count)

    def test_contents(self) -> None:
        expected = {'hugo', 'ilse', 'foo.bar'}
        with mock.patch.object(StorageBackend, '__iter__', return_value=iter(expected)) as iter_mock:
            storage = StorageBackend()
            self.assertEqual(expected, storage.contents)
            self.assertEqual(1, iter_mock.call_count)

    def test_len(self) -> None:
        expected = {'hugo', 'ilse', 'foo.bar'}
        with mock.patch.object(StorageBackend, '__iter__', return_value=iter(expected)) as iter_mock:
            storage = StorageBackend()
            self.assertEqual(3, len(storage))
            self.assertEqual(1, iter_mock.call_count)


class FileSystemBackendTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = TemporaryDirectory()
        self.backend = FilesystemBackend(self.tmp_dir.name)
        self.test_data = 'dshiuasduzchjbfdnbewhsdcuzd'
        self.alternative_testdata = "8u993zhhbn\nb3tadgadg"

        self.identifier = 'some name'

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_init_create_dir(self) -> None:
        path = self.tmp_dir.name + "/inner_dir"
        self.assertFalse(os.path.isdir(path))
        with self.assertRaises(NotADirectoryError):
            FilesystemBackend(path)
            FilesystemBackend(path, create_if_missing=False)
        self.assertFalse(os.path.isdir(path))
        FilesystemBackend(path, create_if_missing=True)
        self.assertTrue(os.path.isdir(path))

    def test_init_file_path(self) -> None:
        with TemporaryFile() as tmp_file:
            with self.assertRaises(NotADirectoryError):
                FilesystemBackend(tmp_file.name)

    def test_put_and_get_normal(self) -> None:
        # first put the data
        self.backend.put(self.identifier, self.test_data)

        # then retrieve it again
        data = self.backend.get(self.identifier)
        self.assertEqual(data, self.test_data)

    def test_put_file_exists_no_overwrite(self) -> None:
        name = 'test_put_file_exists_no_overwrite'
        self.backend.put(name, self.test_data)
        with self.assertRaises(FileExistsError):
            self.backend.put(name, self.alternative_testdata)
        self.assertEqual(self.test_data, self.backend.get(name))

    def test_put_file_exists_overwrite(self) -> None:
        name = 'test_put_file_exists_overwrite'
        self.backend.put(name, self.test_data)
        self.backend.put(name, self.alternative_testdata, overwrite=True)
        self.assertEqual(self.alternative_testdata, self.backend.get(name))

    def test_instantiation_fail(self) -> None:
        with self.assertRaises(NotADirectoryError):
            FilesystemBackend("C\\#~~")

    def test_exists(self) -> None:
        name = 'test_exists'
        self.backend.put(name, self.test_data)
        self.assertTrue(self.backend.exists(name))
        self.assertFalse(self.backend.exists('exists_not'))

    def test_get_not_existing(self) -> None:
        name = 'test_get_not_existing'
        with self.assertRaisesRegex(KeyError, name):
            self.backend.get(name)

    def test_delete(self):
        name = 'test_delete'
        with self.assertRaisesRegex(KeyError, name):
            self.backend.delete(name)

        self.backend.put(name, self.test_data)
        self.assertTrue(self.backend.exists(name))
        self.backend.delete(name)
        self.assertFalse(self.backend.exists(name))
        self.assertFalse(os.listdir(self.tmp_dir.name))

    def test_get_contents_iter_len(self) -> None:
        expected = {'foo', 'bar', 'hugo.test'}
        for name in expected:
            self.backend.put(name, self.test_data)

        self.assertEqual(expected, self.backend.list_contents(), msg="list_contents() faulty")
        self.assertEqual(expected, set(iter(self.backend)), msg="__iter__() faulty")
        self.assertEqual(3, len(self.backend), msg="__len__() faulty")

    def test_iter_empty(self) -> None:
        self.assertEqual(set(), set(iter(self.backend)))


class ZipFileBackendTests(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp_dir = TemporaryDirectory()
        self.path = os.path.join(self.tmp_dir.name, 'backend.zip')
        self.backend = ZipFileBackend(self.path)
        self.assertTrue(zipfile.is_zipfile(self.path))

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_init_invalid_path(self):
        invalid_path = os.path.join(self.tmp_dir.name, "asdfasdf", "backend.zip")
        with self.assertRaises(NotADirectoryError):
            ZipFileBackend(invalid_path)

    def test_init_file_exists_not_zip(self):
        with NamedTemporaryFile() as tmp_file:
            with self.assertRaises(FileExistsError):
                ZipFileBackend(tmp_file.name)

    def test_init_keeps_data(self):
        path = os.path.join(self.tmp_dir.name, 'test.zip')
        with zipfile.ZipFile(path, mode='w', compression=zipfile.ZIP_DEFLATED) as zip_file:
            zip_file.writestr('test_file.txt', 'chichichi')

        ZipFileBackend(path)

        with zipfile.ZipFile(path, 'r') as zip_file:
            ma_string = zip_file.read('test_file.txt')
            self.assertEqual(b'chichichi', ma_string)

    def test_exists(self):
        self.assertFalse(self.backend.exists('foo'))

        with zipfile.ZipFile(self.path, mode='w', compression=zipfile.ZIP_DEFLATED) as zip_file:
            zip_file.writestr('foo.json', 'chichichi')

        self.assertTrue(self.backend.exists('foo'))

    def test_put(self):
        self.backend.put('foo', 'foo_data')

        with zipfile.ZipFile(self.path, 'r') as zip_file:
            ma_string = zip_file.read('foo.json')
            self.assertEqual(b'foo_data', ma_string)

        with self.assertRaises(FileExistsError):
            self.backend.put('foo', 'bar_data')
        with zipfile.ZipFile(self.path, 'r') as zip_file:
            ma_string = zip_file.read('foo.json')
            self.assertEqual(b'foo_data', ma_string)

        self.backend.put('foo', 'foo_bar_data', overwrite=True)
        with zipfile.ZipFile(self.path, 'r') as zip_file:
            ma_string = zip_file.read('foo.json')
            self.assertEqual(b'foo_bar_data', ma_string)

    def test_get(self):
        with self.assertRaises(KeyError):
            self.backend.get('foo')

        data = 'foo_data'
        with zipfile.ZipFile(self.path, 'a') as zip_file:
            zip_file.writestr('foo.json', data)

        self.assertEqual(self.backend.get('foo'), data)

        os.remove(self.path)
        with self.assertRaises(KeyError):
            self.backend.get('foo')

    def test_update(self):
        self.backend.put('foo', 'foo_data')
        self.backend.put('bar', 'bar_data')

        self.backend._update('foo.json', 'foo_bar_data')

        self.assertEqual(self.backend.get('foo'), 'foo_bar_data')
        self.assertEqual(self.backend.get('bar'), 'bar_data')

        self.backend._update('foo.json', None)
        self.assertFalse(self.backend.exists('foo'))

    def test_delete(self):
        with self.assertRaisesRegex(KeyError, 'foo'):
            self.backend.delete('foo')

        self.backend.put('foo', 'foo_data')
        self.assertTrue(self.backend.exists('foo'))
        self.backend.delete('foo')
        self.assertFalse(self.backend.exists('foo'))

        with zipfile.ZipFile(self.path, 'r') as file:
            self.assertNotIn('foo', file.namelist())

    def test_get_contents_iter_len(self) -> None:
        expected = {'foo', 'bar', 'hugo.test'}
        for name in expected:
            self.backend.put(name, "asdfasdfas")

        self.assertEqual(expected, self.backend.list_contents(), msg="list_contents() faulty")
        self.assertEqual(expected, set(iter(self.backend)), msg="__iter__() faulty")
        self.assertEqual(3, len(self.backend), msg="__len__() faulty")

    def test_iter_empty(self) -> None:
        self.assertEqual(set(), set(iter(self.backend)))


class CachingBackendTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dummy_backend = DummyStorageBackend()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
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

    def test_delete(self):
        self.dummy_backend.put('foo', self.testdata)
        self.caching_backend.put('bar', self.alternative_testdata)

        self.caching_backend.delete('bar')
        self.assertNotIn('bar', self.caching_backend)
        self.assertNotIn('bar', self.dummy_backend)

        self.caching_backend.delete('foo')
        self.assertNotIn('foo', self.caching_backend)
        self.assertNotIn('foo', self.dummy_backend)

    def test_get_contents_iter_len(self) -> None:
        expected = {'foo', 'bar', 'hugo.test'}
        for name in expected:
            self.dummy_backend.put(name, "asdfasdfas")

        self.assertEqual(expected, self.caching_backend.list_contents(), msg="list_contents() faulty")
        self.assertEqual(expected, set(iter(self.caching_backend)), msg="__iter__() faulty")
        self.assertEqual(3, len(self.caching_backend), msg="__len__() faulty")

    def test_iter_empty(self) -> None:
        self.assertEqual(set(), set(iter(self.caching_backend)))


class DictBackendTests(unittest.TestCase):
    def setUp(self):
        self.backend = DictBackend()

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

    def test_delete(self):
        self.backend.put('a', 'data')

        with self.assertRaises(KeyError):
            self.backend.delete('b')

        self.backend.delete('a')
        self.assertFalse(self.backend.storage)
        
    def test_get_contents_iter_len(self) -> None:
        expected = {'foo', 'bar', 'hugo.test'}
        for name in expected:
            self.backend.put(name, "asdfasdfas")

        self.assertEqual(expected, self.backend.list_contents(), msg="list_contents() faulty")
        self.assertEqual(expected, set(iter(self.backend)), msg="__iter__() faulty")
        self.assertEqual(3, len(self.backend), msg="__len__() faulty")

    def test_iter_empty(self) -> None:
        self.assertEqual(set(), set(iter(self.backend)))
        

class DeserializationCallbackFinderTests(unittest.TestCase):
    def test_set_item(self):
        finder = DeserializationCallbackFinder()

        def my_callable():
            pass

        finder['asd'] = my_callable

        self.assertIn('asd', finder)

    def test_auto_import(self):
        finder = DeserializationCallbackFinder()

        with mock.patch('importlib.import_module') as import_module, mock.patch.dict(sys.modules, clear=True):
            with self.assertRaises(KeyError):
                _ = finder['qupulse.pulses.table_pulse_template.TablePulseTemplate']
            import_module.assert_called_once_with('qupulse.pulses.table_pulse_template')

        finder.auto_import = False
        with mock.patch('importlib.import_module') as import_module, mock.patch.dict(sys.modules, clear=True):
            with self.assertRaises(KeyError):
                finder['qupulse.pulses.table_pulse_template.TablePulseTemplate']
            import_module.assert_not_called()

    def test_qctoolkit_import(self):
        def my_callable():
            pass

        finder = DeserializationCallbackFinder()

        finder['qupulse.asd'] = my_callable
        self.assertIs(finder['qctoolkit.asd'], my_callable)

        finder.qctoolkit_alias = False
        with self.assertRaises(KeyError):
            finder['qctoolkit.asd']


class SerializableMetaTests(unittest.TestCase):
    def test_native_deserializable(self):

        class NativeDeserializable(metaclass=SerializableMeta):
            @classmethod
            def get_type_identifier(cls):
                return 'foo.bar.never'

        self.assertIn('foo.bar.never', SerializableMeta.deserialization_callbacks)
        self.assertEqual(SerializableMeta.deserialization_callbacks['foo.bar.never'], NativeDeserializable)


class DefaultPulseRegistryManipulationTests(unittest.TestCase):

    def test_get_set_default_pulse_registry(self) -> None:
        # store previous registry
        previous_registry = get_default_pulse_registry()

        registry = dict()
        set_default_pulse_registry(registry)
        self.assertIs(get_default_pulse_registry(), registry)

        # restore previous registry
        set_default_pulse_registry(previous_registry)
        self.assertIs(get_default_pulse_registry(), previous_registry)

    def test_new_default_pulse_registry(self) -> None:
        # store previous registry
        previous_registry = get_default_pulse_registry()

        new_default_pulse_registry()
        self.assertIsNotNone(get_default_pulse_registry())
        self.assertIsNot(get_default_pulse_registry(), previous_registry)

        # restore previous registry
        set_default_pulse_registry(previous_registry)
        self.assertIs(get_default_pulse_registry(), previous_registry)


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
        instance_1 = DummySerializable(identifier='my_id', registry=dict())
        instance_2 = DummySerializable(identifier='my_id', registry=dict())

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

    def test_setitem_different_id(self) -> None:
        serializable = DummySerializable(identifier='my_id', registry=dict())
        with self.assertRaisesRegex(ValueError, "different than its own internal identifier"):
            self.storage['a_totally_different_id'] = serializable

    def test_setitem_duplicate_only_in_backend(self) -> None:
        serializable = DummySerializable(identifier='my_id', registry=dict())
        backend = DummyStorageBackend()
        backend['my_id'] = 'data_in_storage'
        storage = PulseStorage(backend)
        with self.assertRaisesRegex(RuntimeError, "assigned in storage backend"):
            storage['my_id'] = serializable
        self.assertEqual({'my_id': 'data_in_storage'}, backend.stored_items)

    def test_overwrite(self):

        encode_mock = mock.Mock(return_value='asd')

        instance = DummySerializable(identifier='my_id')

        with mock.patch.object(JSONSerializableEncoder, 'encode', new=encode_mock):
            self.storage.overwrite('my_id', instance)

        self.assertEqual(encode_mock.call_count, 1)
        self.assertEqual(encode_mock.call_args, mock.call(instance.get_serialization_data()))

        self.assertEqual(self.storage._temporary_storage, {'my_id': self.storage.StorageEntry('asd', instance)})

    def test_write_through(self):
        instance_1 = DummySerializable(identifier='my_id_1', registry=dict())
        inner_instance = DummySerializable(identifier='my_id_2', registry=dict())
        outer_instance = DummySerializable(inner=inner_instance, identifier='my_id_3', registry=dict())

        def get_expected():
            return {identifier: serialized
                    for identifier, (serialized, _) in self.storage.temporary_storage.items()}

        self.storage['my_id_1'] = instance_1
        self.storage['my_id_3'] = outer_instance

        self.assertEqual(get_expected(), self.backend.stored_items)

    def test_write_through_does_not_overwrite_subpulses(self) -> None:
        previous_inner = DummySerializable(identifier='my_id_1', data='hey', registry=dict())
        inner_instance = DummySerializable(identifier='my_id_1', data='ho', registry=dict())
        outer_instance = DummySerializable(inner=inner_instance, identifier='my_id_2', registry=dict())

        self.storage['my_id_1'] = previous_inner
        with self.assertRaises(RuntimeError):
            self.storage['my_id_2'] = outer_instance
        self.assertNotIn('my_id_2', self.storage)
        self.assertNotIn('my_id_2', self.backend)
        self.assertIs(previous_inner, self.storage['my_id_1'])

        enc = JSONSerializableEncoder(None, sort_keys=True, indent=4)
        expected = enc.encode(previous_inner.get_serialization_data())
        self.assertEqual(expected, self.backend['my_id_1'])

    def test_failed_overwrite_does_not_leave_subpulses(self) -> None:
        inner_named = DummySerializable(data='bar', identifier='inner')
        inner_known = DummySerializable(data='bar', identifier='known', registry=dict())
        outer = DummySerializable(data=[inner_named, inner_known], identifier='outer')
        inner_known_previous = DummySerializable(data='b38azodhg', identifier='known', registry=dict())

        self.storage['known'] = inner_known_previous

        self.assertIn('known', self.storage)
        with self.assertRaises(RuntimeError):
            self.storage['outer'] = outer

        self.assertNotIn('outer', self.storage)
        self.assertNotIn('inner', self.storage)

    def test_clear(self):
        instance_1 = DummySerializable(identifier='my_id_1')
        instance_2 = DummySerializable(identifier='my_id_2')
        instance_3 = DummySerializable(identifier='my_id_3')

        self.storage['my_id_1'] = instance_1
        self.storage['my_id_2'] = instance_2
        self.storage['my_id_3'] = instance_3

        self.storage.clear()

        self.assertFalse(self.storage.temporary_storage)

    def test_as_default_registry(self) -> None:
        prev_reg = get_default_pulse_registry()
        pulse_storage = PulseStorage(DummyStorageBackend())
        with pulse_storage.as_default_registry():
            self.assertIs(get_default_pulse_registry(), pulse_storage)
        self.assertIs(get_default_pulse_registry(), prev_reg)

    def test_set_to_default_registry(self) -> None:
        pulse_storage = PulseStorage(DummyStorageBackend())
        previous_default_registry = get_default_pulse_registry()
        try:
            pulse_storage.set_to_default_registry()
            self.assertIs(get_default_pulse_registry(), pulse_storage)
        finally:
            set_default_pulse_registry(previous_default_registry)

    def test_beautified_json(self) -> None:
        data = {'e': 89, 'b': 151, 'c': 123515, 'a': 123, 'h': 2415}
        template = DummySerializable(data=data, identifier="foo")
        pulse_storage = PulseStorage(DummyStorageBackend())
        pulse_storage['foo'] = template

        expected = """{
    \"#identifier\": \"foo\",
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

    def test_delitem(self):
        instance_1 = DummySerializable(identifier='my_id_1')
        instance_2 = DummySerializable(identifier='my_id_2')

        backend = DummyStorageBackend()

        pulse_storage = PulseStorage(backend)
        with self.assertRaises(KeyError):
            del pulse_storage[instance_1.identifier]

        # write first instance to backend
        pulse_storage[instance_1.identifier] = instance_1

        del pulse_storage

        # now instance_1 is not in the temporary storage
        pulse_storage = PulseStorage(backend)
        pulse_storage[instance_2.identifier] = instance_2

        del pulse_storage[instance_1.identifier]
        del pulse_storage[instance_2.identifier]

        self.assertEqual({}, backend.stored_items)
        self.assertEqual(pulse_storage.temporary_storage, {})

    @mock.patch.multiple(StorageBackend, __abstractmethods__=set())
    def test_len(self) -> None:
        with mock.patch.object(StorageBackend, '__len__', return_value=5) as len_mock:
            pulse_storage = PulseStorage(StorageBackend())
            self.assertEqual(5, len(pulse_storage))
            self.assertEqual(1, len_mock.call_count)

    @mock.patch.multiple(StorageBackend, __abstractmethods__=set())
    def test_iter(self) -> None:
        data = ['hugo', 'ilse', 'foo.bar']
        with mock.patch.object(StorageBackend, '__iter__', return_value=iter(data)) as iter_mock:
            pulse_storage = PulseStorage(StorageBackend())
            self.assertEqual(set(data), set(iter(pulse_storage)))
            self.assertEqual(1, iter_mock.call_count)

    @mock.patch.multiple(StorageBackend, __abstractmethods__=set())
    def test_contents(self) -> None:
        data = ['hugo', 'ilse', 'foo.bar']
        with mock.patch.object(StorageBackend, '__iter__', return_value=iter(data)) as iter_mock:
            pulse_storage = PulseStorage(StorageBackend())
            self.assertEqual(set(data), set(iter(pulse_storage)))
            self.assertEqual(1, iter_mock.call_count)

    def test_deserialize_storage_is_default_registry(self) -> None:
        backend = DummyStorageBackend()

        # fill backend
        serializable = DummySerializable(identifier='peter', registry=dict())
        pulse_storage = PulseStorage(backend)
        pulse_storage['peter'] = serializable
        del pulse_storage

        # try to deserialize while PulseStorage is default registry
        pulse_storage = PulseStorage(backend)
        with pulse_storage.as_default_registry():
            deserialized = pulse_storage['peter']
            self.assertEqual(deserialized, serializable)

    def test_deserialize_storage_is_not_default_registry_id_free(self) -> None:
        backend = DummyStorageBackend()

        # fill backend
        serializable = DummySerializable(identifier='peter', registry=dict())
        pulse_storage = PulseStorage(backend)
        pulse_storage['peter'] = serializable
        del pulse_storage

        pulse_storage = PulseStorage(backend)
        deserialized = pulse_storage['peter']
        self.assertEqual(deserialized, serializable)

    @unittest.mock.patch('qupulse.serialization.default_pulse_registry', dict())
    def test_deserialize_storage_is_not_default_registry_id_occupied(self) -> None:
        backend = DummyStorageBackend()

        # fill backend
        serializable = DummySerializable(identifier='peter')
        pulse_storage = PulseStorage(backend)
        pulse_storage['peter'] = serializable
        del pulse_storage

        pulse_storage = PulseStorage(backend)
        with self.assertRaisesRegex(RuntimeError, "Pulse with name already exists"):
            pulse_storage['peter']

    def test_deserialize_twice_same_object_storage_is_default_registry(self) -> None:
        backend = DummyStorageBackend()

        # fill backend
        serializable = DummySerializable(identifier='peter', registry=dict())
        pulse_storage = PulseStorage(backend)
        pulse_storage['peter'] = serializable
        del pulse_storage

        # try to deserialize while PulseStorage is default registry
        pulse_storage = PulseStorage(backend)
        with pulse_storage.as_default_registry():
            deserialized_1 = pulse_storage['peter']
            deserialized_2 = pulse_storage['peter']
            self.assertIs(deserialized_1, deserialized_2)
            self.assertEqual(deserialized_1, serializable)

    @unittest.mock.patch('qupulse.serialization.default_pulse_registry', None)
    def test_consistent_over_instances(self) -> None:
        # tests that PulseStorage behaves consistently over several instance (especially with regards to duplicate test)
        # demonstrates issue #273
        identifier = 'hugo'
        hidden_serializable = DummySerializable(identifier=identifier, foo='bar')
        serializable = DummySerializable(identifier=identifier, data={'abc': 123, 'cde': 'fgh'})

        backend = DummyStorageBackend()

        pulse_storage = PulseStorage(backend)
        pulse_storage[identifier] = hidden_serializable
        with self.assertRaises(RuntimeError):
            pulse_storage[identifier] = serializable
        deserialized = pulse_storage[serializable.identifier]
        self.assertEqual(hidden_serializable, deserialized)

        pulse_storage = PulseStorage(backend)
        with self.assertRaises(RuntimeError):
            pulse_storage[serializable.identifier] = serializable
        deserialized = pulse_storage[serializable.identifier]
        self.assertEqual(hidden_serializable, deserialized)

        pulse_storage = PulseStorage(backend)
        deserialized = pulse_storage[serializable.identifier]
        self.assertEqual(hidden_serializable, deserialized)


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

    def test_default_else_branch(self) -> None:
        encoder = JSONSerializableEncoder(dict())
        data = {'a': 'bc', 'b': [1, 2, 3]}

        with self.assertRaises(TypeError):
            encoder.default(data)

    def test_encoding(self):
        class A(AnonymousSerializable):
            anonymous_serialization_data = [1, 2, 3]

            def get_serialization_data(self):
                return self.anonymous_serialization_data

        inner_anon = DummySerializable(data=[A(), 1])
        inner_named = DummySerializable(data='bar', identifier='inner')
        inner_known = DummySerializable(data='bar', identifier='known')

        outer = DummySerializable(data=[inner_named, inner_anon, inner_known])

        storage = dict(known=inner_known)
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
        self.assertIs(storage['known'], inner_known)

    def test_encoding_duplicated_id(self):
        inner_named = DummySerializable(data='bar', identifier='inner', registry=dict())
        inner_known = DummySerializable(data='bar', identifier='known', registry=dict())
        inner_known_previous = DummySerializable(data='abh3h8ga', identifier='known', registry=dict())

        outer = DummySerializable(data=[inner_named, inner_known])

        storage = dict(known=inner_known_previous)
        encoder = JSONSerializableEncoder(storage)

        with self.assertRaises(RuntimeError):
            encoder.encode(outer)

        self.assertEqual(set(storage.keys()), {'inner', 'known'})
        self.assertIs(storage['inner'], inner_named)
        self.assertIs(storage['known'], inner_known_previous)

########################################################################################################################
################################ tests for old architecture, now deprecated ############################################
########################################################################################################################

import warnings
from typing import Dict
from qupulse.serialization import ExtendedJSONEncoder, Serializer
from qupulse.pulses.table_pulse_template import TablePulseTemplate
from qupulse.pulses.sequence_pulse_template import SequencePulseTemplate


class NestedDummySerializable(Serializable):

    def __init__(self, data: Serializable, identifier: Optional[str]=None, registry: PulseRegistryType=None) -> None:
        super().__init__(identifier)
        self.data = data
        self._register(registry=registry)

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
        serializable = DummySerializable(identifier='bar', registry=dict())
        serialized = self.serializer.dictify(serializable)
        self.assertEqual(serializable.identifier, serialized)

    def test_serialize_subpulse_duplicate_identifier(self) -> None:
        serializable = DummySerializable(identifier='bar', registry=dict())
        self.serializer.dictify(serializable)
        self.serializer.dictify(serializable)
        serializable = DummySerializable(data='this is other data than before', identifier='bar', registry=dict())
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
        registry = dict()
        table_foo = TablePulseTemplate(identifier='foo', entries={'default': [('hugo', 2),
                                                                              ('albert', 'voltage')]},
                                       parameter_constraints=['albert<9.1'],
                                       registry=registry)
        table = TablePulseTemplate({'default': [('t', 0)]})

        foo_mappings = dict(hugo='ilse', albert='albert', voltage='voltage')
        sequence = SequencePulseTemplate((table_foo, foo_mappings, dict()),
                                         (table, dict(t=0), dict()),
                                         identifier=None,
                                         registry=registry)
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


# the following are tests for the routines that convert pulses from old to new serialization formats
# can be removed after transition period
# todo (218-06-14): remove ConversionTests after finalizing transition period from old to new serialization routines
from qupulse.serialization import convert_stored_pulse_in_storage, convert_pulses_in_storage


class ConversionTests(unittest.TestCase):

    def test_convert_stored_pulse_in_storage(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)

            source_backend = DummyStorageBackend()
            serializer = Serializer(source_backend)

            hugo_serializable = DummySerializable(foo='bar',
                                                  identifier='hugo',
                                                  registry=dict())

            serializable = NestedDummySerializable(hugo_serializable, identifier='hugos_parent', registry=dict())
            serializer.serialize(serializable)

            destination_backend = DummyStorageBackend()
            convert_stored_pulse_in_storage('hugos_parent', source_backend, destination_backend)

            pulse_storage = PulseStorage(destination_backend)
            deserialized = pulse_storage['hugos_parent']
            self.assertEqual(serializable, deserialized)

    def test_convert_stored_pulse_in_storage_dest_not_empty_id_overlap(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)

            source_backend = DummyStorageBackend()
            serializer = Serializer(source_backend)

            hugo_serializable = DummySerializable(foo='bar',
                                                  identifier='hugo',
                                                  registry=dict())

            serializable = NestedDummySerializable(hugo_serializable, identifier='hugos_parent', registry=dict())
            serializer.serialize(serializable)

            destination_backend = DummyStorageBackend()
            destination_backend.put('hugo', 'already_existing_data')
            with self.assertRaises(ValueError):
                convert_stored_pulse_in_storage('hugos_parent', source_backend, destination_backend)

            self.assertEquals('already_existing_data', destination_backend['hugo'])
            self.assertEquals(1, len(destination_backend.stored_items))

    def test_convert_stored_pulse_in_storage_dest_not_empty_no_id_overlap(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)

            source_backend = DummyStorageBackend()
            serializer = Serializer(source_backend)

            hugo_serializable = DummySerializable(foo='bar',
                                                  identifier='hugo',
                                                  registry=dict())

            serializable = NestedDummySerializable(hugo_serializable, identifier='hugos_parent', registry=dict())
            serializer.serialize(serializable)

            destination_backend = DummyStorageBackend()
            destination_backend.put('ilse', 'already_existing_data')
            convert_stored_pulse_in_storage('hugos_parent', source_backend, destination_backend)

            self.assertEquals('already_existing_data', destination_backend['ilse'])
            pulse_storage = PulseStorage(destination_backend)
            deserialized = pulse_storage['hugos_parent']
            self.assertEqual(serializable, deserialized)

    def test_convert_stored_pulses(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)

            source_backend = DummyStorageBackend()
            serializer = Serializer(source_backend)

            hugo_serializable = DummySerializable(foo='bar',
                                                  identifier='hugo',
                                                  registry=dict())

            serializable_a = NestedDummySerializable(hugo_serializable, identifier='hugos_parent', registry=dict())
            serializable_b = DummySerializable(identifier='ilse',
                                               foo=dict(abc=123, data='adf8g23'),
                                               number=7.3,
                                               registry=dict())

            serializer.serialize(serializable_a)
            serializer.serialize(serializable_b)

            destination_backend = DummyStorageBackend()
            convert_pulses_in_storage(source_backend, destination_backend)

            pulse_storage = PulseStorage(destination_backend)
            deserialized_a = pulse_storage['hugos_parent']
            deserialized_b = pulse_storage['ilse']
            self.assertEqual(serializable_a, deserialized_a)
            self.assertEqual(serializable_b, deserialized_b)

    def test_convert_stored_pulses_dest_not_empty_id_overlap(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)

            source_backend = DummyStorageBackend()
            serializer = Serializer(source_backend)

            hugo_serializable = DummySerializable(foo='bar',
                                                  identifier='hugo',
                                                  registry=dict())

            serializable_a = NestedDummySerializable(hugo_serializable, identifier='hugos_parent', registry=dict())
            serializable_b = DummySerializable(identifier='ilse',
                                               foo=dict(abc=123, data='adf8g23'),
                                               number=7.3,
                                               registry=dict())

            serializer.serialize(serializable_a)
            serializer.serialize(serializable_b)

            destination_backend = DummyStorageBackend()
            destination_backend.put('hugo', 'already_existing_data')
            with self.assertRaises(ValueError):
                convert_pulses_in_storage(source_backend, destination_backend)

            self.assertEquals('already_existing_data', destination_backend['hugo'])
            self.assertEquals(1, len(destination_backend.stored_items))

    def test_convert_stored_pulses_dest_not_empty_no_id_overlap(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)

            source_backend = DummyStorageBackend()
            serializer = Serializer(source_backend)

            hugo_serializable = DummySerializable(foo='bar',
                                                  identifier='hugo',
                                                  registry=dict())

            serializable_a = NestedDummySerializable(hugo_serializable, identifier='hugos_parent', registry=dict())
            serializable_b = DummySerializable(identifier='ilse',
                                               foo=dict(abc=123, data='adf8g23'),
                                               number=7.3,
                                               registry=dict())

            serializer.serialize(serializable_a)
            serializer.serialize(serializable_b)

            destination_backend = DummyStorageBackend()
            destination_backend.put('peter', 'already_existing_data')
            convert_pulses_in_storage(source_backend, destination_backend)

            self.assertEqual('already_existing_data', destination_backend['peter'])
            pulse_storage = PulseStorage(destination_backend)
            deserialized_a = pulse_storage['hugos_parent']
            deserialized_b = pulse_storage['ilse']
            self.assertEqual(serializable_a, deserialized_a)
            self.assertEqual(serializable_b, deserialized_b)
