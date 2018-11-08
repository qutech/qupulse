""" This module provides serialization and storage functionality.

Classes:
    - StorageBackend: Abstract representation of a data storage.
    - FilesystemBackend: Implementation of a file system data storage.
    - ZipFileBackend: Like FilesystemBackend but inside a single zip file instead of a directory
    - CachingBackend: A caching decorator for StorageBackends.
    - Serializable: An interface for serializable objects.
    - PulseStorage: High-level management object for loading and storing and transparently (de)serializing serializable objects.

Deprecated Classes:
    - Serializer: Converts Serializables to a serial representation as a string and vice-versa.

Functions:
    - get_default_pulse_registry: Returns the default pulse registry
    - set_default_pulse_registry: Set the default pulse registry
    - new_default_pulse_registry: Reset the default pulse registry with an empty mapping
    - convert_stored_pulse_in_storage: Converts a single Serializable stored using the deprecated Serializer class format into the PulseStorage format.
    - convert_pulses_in_storage: Converts all Serializables stored in a StorageBackend using the deprecated Serializer class format into the PulseStorage format.
"""

from abc import ABCMeta, abstractmethod
from typing import Dict, Any, Optional, NamedTuple, Union, Mapping, MutableMapping, Set, Callable, Iterator, Iterable
import os
import zipfile
import tempfile
import json
import weakref
import warnings
import gc
import importlib
import warnings
from contextlib import contextmanager

from qupulse.utils.types import DocStringABCMeta

__all__ = ["StorageBackend", "FilesystemBackend", "ZipFileBackend", "CachingBackend", "Serializable", "Serializer",
           "AnonymousSerializable", "DictBackend", "PulseStorage",
           "convert_pulses_in_storage", "convert_stored_pulse_in_storage", "PulseRegistryType", "get_default_pulse_registry",
           "set_default_pulse_registry", "new_default_pulse_registry"]


class StorageBackend(metaclass=ABCMeta):
    """A backend to store data/files in.

    Used as an abstraction of file systems/databases for the serializer.
    """

    @abstractmethod
    def put(self, identifier: str, data: str, overwrite: bool=False) -> None:
        """Stores the data string identified by identifier.

        Args:
            identifier (str): A unique identifier/name for the data to be stored.
            data (str): A serialized string of data to be stored.
            overwrite (bool): Set to True, if already existing data shall be overwritten.
                (default: False)
        Raises:
            FileExistsError if overwrite is False and there already exists data which
                is associated with the given identifier.
        """

    def __setitem__(self, identifier: str, data: str) -> None:
        self.put(identifier, data)

    @abstractmethod
    def get(self, identifier: str) -> str:
        """Retrieves the data string with the given identifier.

        Args:
            identifier (str): The identifier of the data to be retrieved.
        Returns:
            A serialized string of the data associated with the given identifier, if present.
        Raises:
            KeyError if no data is associated with the given identifier.
        """

    def __getitem__(self, identifier: str) -> str:
        return self.get(identifier)

    @abstractmethod
    def exists(self, identifier: str) -> bool:
        """Checks if data is stored for the given identifier.

        Args:
            identifier (str): The identifier for which presence of data shall be checked.
        Returns:
            True, if stored data is associated with the given identifier.
        """

    def __contains__(self, identifier: str) -> bool:
        return self.exists(identifier)

    @abstractmethod
    def delete(self, identifier: str) -> None:
        """Deletes data of the given identifier.

        Args:
            identifier: identifier of the data to be deleted

        Raises:
            KeyError if there is no data associated with the identifier
        """

    def __delitem__(self, identifier: str) -> None:
        self.delete(identifier)

    def list_contents(self) -> Iterable[str]:
        """Returns a listing of all available identifiers.

        DEPRECATED (2018-09-20): Use property contents instead.

        Returns:
            List of all available identifiers.
        """
        warnings.warn("list_contents is deprecated. Use the property contents instead", DeprecationWarning)
        return self.contents

    @property
    def contents(self) -> Iterable[str]:
        """The identifiers of all Serializables currently stored in this StorageBackend."""
        return set(iter(self))

    @abstractmethod
    def __iter__(self) -> Iterator[str]:
        """Iterator over all identifiers of Serializables currently stored in this StorageBackend."""
        pass

    def __len__(self) -> int:
        return len(self.contents)


class FilesystemBackend(StorageBackend):
    """A StorageBackend implementation based on a regular filesystem.

    Data will be stored in plain text files in a directory. The directory is given in the
    constructor of this FilesystemBackend. For each data item, a separate file is created an named
    after the corresponding identifier. If the directory does not exist, it is not created unless the create_if_missing
    argument is explicitly set to True.
    """

    def __init__(self, root: str='.', create_if_missing: bool=False) -> None:
        """Creates a new FilesystemBackend.

        Args:
            root: The path of the directory in which all data files are located. (default: ".",
                i.e. the current directory)
            create_if_missing: If False, do not create the specified directory if it does not exist. (default: False)
        Raises:
            NotADirectoryError: if root is not a valid directory path.
        """
        if not os.path.exists(root) and create_if_missing:
            os.makedirs(root)
        if not os.path.isdir(root):
            raise NotADirectoryError()
        self._root = os.path.abspath(root)

    def _path(self, identifier) -> str:
        return os.path.join(self._root, identifier + '.json')

    def put(self, identifier: str, data: str, overwrite: bool=False) -> None:
        if self.exists(identifier) and not overwrite:
            raise FileExistsError(identifier)
        path = self._path(identifier)
        with open(path, 'w') as file:
            file.write(data)

    def get(self, identifier: str) -> str:
        path = self._path(identifier)
        try:
            with open(path) as file:
                return file.read()
        except FileNotFoundError as fnf:
            raise KeyError(identifier) from fnf

    def exists(self, identifier: str) -> bool:
        path = self._path(identifier)
        return os.path.isfile(path)

    def delete(self, identifier):
        try:
            os.remove(self._path(identifier))
        except FileNotFoundError as fnf:
            raise KeyError(identifier) from fnf

    def __iter__(self) -> Iterator[str]:
        for dirpath, dirs, files in os.walk(self._root):
            return (filename for filename, ext in (os.path.splitext(file) for file in files) if ext == '.json')


class ZipFileBackend(StorageBackend):
    """A StorageBackend implementation based on a single zip file.

    Data will be stored in plain text files inside a zip file. The zip file is given
    in the constructor of this FilesystemBackend. For each data item, a separate
    file is created and named after the corresponding identifier.

    ZipFileBackend uses significantly less storage space and is faster on
    network devices, but takes longer to update because every write causes a
    complete recompression (it's not too bad)."""

    def __init__(self, root: str='./storage.zip', compression_method: int=zipfile.ZIP_DEFLATED) -> None:
        """Creates a new FilesystemBackend.

        Args:
            root: The path of the zip file in which all data files are stored. (default: "./storage.zip",
                i.e. the current directory)
            compression_method: The compression method/algorithm used to compress data in the zipfile. Accepts
                all values handled by the zipfile module (ZIP_STORED, ZIP_DEFLATED, ZIP_BZIP2, ZIP_LZMA). Please refer
                to the `zipfile docs <https://docs.python.org/3/library/zipfile.html#zipfile.ZIP_STORED>` for more
                information. (default: zipfile.ZIP_DEFLATED)
        Raises:
            NotADirectoryError if root is not a valid path.
        """
        parent, fname = os.path.split(root)
        if not os.path.isfile(root):
            if not os.path.isdir(parent):
                raise NotADirectoryError(
                    "Cannot create a ZipStorageBackend. The parent path {} is not valid.".format(parent)
                )
            z = zipfile.ZipFile(root, "w")
            z.close()
        elif not zipfile.is_zipfile(root):
            raise FileExistsError("Cannot open a ZipStorageBackend. The file {} is not a zip archive.".format(root))
        self._root = root
        self._compression_method = compression_method

    def _path(self, identifier) -> str:
        return os.path.join(identifier + '.json')

    def put(self, identifier: str, data: str, overwrite: bool=False) -> None:
        if not self.exists(identifier):
            with zipfile.ZipFile(self._root, mode='a', compression=self._compression_method) as myzip:
                path = self._path(identifier)
                myzip.writestr(path, data)
        else:
            if overwrite:
                self._update(self._path(identifier), data)
            else:
                raise FileExistsError(identifier)

    def get(self, identifier: str) -> str:
        path = self._path(identifier)
        try:
            with zipfile.ZipFile(self._root) as myzip:
                with myzip.open(path) as file:
                    return file.read().decode()
        except FileNotFoundError as fnf:
            raise KeyError(identifier) from fnf

    def exists(self, identifier: str) -> bool:
        path = self._path(identifier)
        with zipfile.ZipFile(self._root, 'r') as myzip:
            return path in myzip.namelist()

    def delete(self, identifier: str) -> None:
        if not self.exists(identifier):
            raise KeyError(identifier)
        self._update(self._path(identifier), None)

    def _update(self, filename: str, data: Optional[str]) -> None:
        # generate a temp file
        tmpfd, tmpname = tempfile.mkstemp(dir=os.path.dirname(self._root))
        os.close(tmpfd)

        # create a temp copy of the archive without filename            
        with zipfile.ZipFile(self._root, 'r') as zin:
            with zipfile.ZipFile(tmpname, 'w') as zout:
                zout.comment = zin.comment # preserve the comment
                for item in zin.infolist():
                    if item.filename != filename:
                        zout.writestr(item, zin.read(item.filename))

        # replace with the temp archive
        os.remove(self._root)
        os.rename(tmpname, self._root)

        # now add filename with its new data
        if data is not None:
            with zipfile.ZipFile(self._root, mode='a', compression=self._compression_method) as zf:
                zf.writestr(filename, data)

    def __iter__(self) -> Iterator[str]:
        with zipfile.ZipFile(self._root, 'r') as myzip:
            return (filename
                    for filename, ext in (os.path.splitext(file) for file in myzip.namelist())
                    if ext == '.json')


class CachingBackend(StorageBackend):
    """Adds naive memory caching functionality to another StorageBackend.

    CachingBackend relies on another StorageBackend to provide real data IO functionality which
    it extends by caching already opened files in memory for faster subsequent access.

    Note that it does not automatically clear the cache at any time and thus will consume increasing amounts of memory
    over time. Use the :meth:`clear_cache` method to clear the cache manually.

    DEPRECATED (2018-09-20): PulseStorage now already provides chaching around StorageBackends, rendering CachingBackend
    obsolete.
    """

    def __init__(self, backend: StorageBackend) -> None:
        """Creates a new CachingBackend.

        Args:
            backend (StorageBackend): A StorageBackend that provides data
                IO functionality.
        """
        warnings.warn("CachingBackend is obsolete due to PulseStorage already offering caching functionality.",
                      DeprecationWarning)
        self._backend = backend
        self._cache = {}

    def put(self, identifier: str, data: str, overwrite: bool=False) -> None:
        if identifier in self._cache and not overwrite:
            raise FileExistsError(identifier)
        self._backend.put(identifier, data, overwrite)
        self._cache[identifier] = data

    def get(self, identifier: str) -> str:
        if identifier not in self._cache:
            self._cache[identifier] = self._backend.get(identifier)
        return self._cache[identifier]

    def exists(self, identifier: str) -> bool:
        return self._backend.exists(identifier)

    def delete(self, identifier: str) -> None:
        self._backend.delete(identifier)
        if identifier in self._cache:
            del self._cache[identifier]

    def __iter__(self) -> Iterator[str]:
        return iter(self._backend)

    def clear_cache(self) -> None:
        self._cache = dict()


class DictBackend(StorageBackend):
    """DictBackend uses a dictionary to store Serializables in memory.

    Doing so, it does not provide any persistent storage functionality.
    """
    def __init__(self) -> None:
        self._cache = {}

    def put(self, identifier: str, data: str, overwrite: bool=False) -> None:
        if identifier in self._cache and not overwrite:
            raise FileExistsError(identifier)
        self._cache[identifier] = data

    def get(self, identifier: str) -> str:
        return self._cache[identifier]

    def exists(self, identifier: str) -> bool:
        return identifier in self._cache
    
    @property
    def storage(self) -> Dict[str, str]:
        return self._cache

    def delete(self, identifier: str) -> None:
        del self._cache[identifier]

    def __iter__(self) -> Iterator[str]:
        return iter(self._cache)


class DeserializationCallbackFinder:
    def __init__(self):
        self._storage = {}
        self.auto_import = True
        self.qctoolkit_alias = True

    def __setitem__(self, type_name: str, callback: Callable):
        self._storage[type_name] = callback

    def __getitem__(self, type_name: str) -> Callable:
        if self.qctoolkit_alias and type_name.startswith('qctoolkit.'):
            type_name = type_name.replace('qctoolkit.', 'qupulse.')

        if self.auto_import and type_name not in self._storage:
            module_name = '.'.join(type_name.split('.')[:-1])
            importlib.import_module(module_name)

        return self._storage[type_name]

    def __contains__(self, type_name) -> bool:
        return type_name in self._storage


class SerializableMeta(DocStringABCMeta):
    deserialization_callbacks = DeserializationCallbackFinder()

    def __new__(mcs, name, bases, dct):
        cls = super().__new__(mcs, name, bases, dct)

        type_identifier = getattr(cls, 'get_type_identifier')()

        try:
            deserialization_function = getattr(cls, 'deserialize')
        except AttributeError:
            deserialization_function = cls
        mcs.deserialization_callbacks[type_identifier] = deserialization_function

        return cls


PulseRegistryType = Optional[MutableMapping[str, 'Serializable']]
default_pulse_registry = None # type: PulseRegistryType


def get_default_pulse_registry() -> PulseRegistryType:
    """Returns the current default pulse registry."""
    return default_pulse_registry


def set_default_pulse_registry(new_default_registry: PulseRegistryType) -> None:
    """Sets the default pulse registry.

    Args:
        new_default_registry: Any PulseRegistryType  object (i.e., mutable mapping) which will become the new default
            pulse registry.
    """
    global default_pulse_registry
    default_pulse_registry = new_default_registry


def new_default_pulse_registry() -> None:
    """Sets a new empty default pulse registry.

    The new registry is a newly created weakref.WeakValueDictionry().
    """
    set_default_pulse_registry(weakref.WeakValueDictionary())


class Serializable(metaclass=SerializableMeta):
    """Any object that can be converted into a serialized representation for storage and back.

    Serializable is the interface used by PulseStorage to obtain representations of objects that
    need to be stored. It essentially provides the methods get_serialization_data, which returns
    a dictionary which contains all relevant properties of the Serializable object encoded as
    basic Python types, and deserialize, which is able to reconstruct the object from given
    such a dictionary.

    Additionally, a Serializable object MAY have a unique identifier, which indicates towards
    the PulseStorage that this object should be stored as a separate data item and accessed by
    reference instead of possibly embedding it into a containing Serializable's representation.

    All Serializables MUST automatically register themselves with the default pulse registry on
    construction unless an explicit other registry is provided to them as construction argument.
    This MUST be implemented by all subclasses of Serializable by calling `Serializable._register` at some point
    in their __init__ method.
    This is intended to prevent accidental duplicate usage of identifiers by failing early.

    See also:
        PulseStorage
    """

    type_identifier_name = '#type'
    identifier_name = '#identifier'

    def __init__(self, identifier: Optional[str]=None) -> None:
        """Initializes a Serializable.

        Args:
            identifier: An optional, non-empty identifier for this Serializable.
                If set, this Serializable will always be stored as a separate data item and never
                be embedded.
        Raises:
            ValueError: If identifier is the empty string
        """
        super().__init__()

        if identifier == '':
            raise ValueError("Identifier must not be empty.")
        self.__identifier = identifier

    def _register(self, registry: Optional[PulseRegistryType]=None) -> None:
        """Registers the Serializable in the global registry.

        This method MUST be called by subclasses at some point during init.
        Args:
            registry: An optional mutable mapping where the Serializable is registered. If None, it gets registered in
                the default_pulse_registry.
        Raises:
            RuntimeError: If a Serializable with the same name is already registered.
        """
        if registry is None:
            registry = default_pulse_registry

        if self.identifier and registry is not None:
            if self.identifier in registry and isinstance(registry, weakref.WeakValueDictionary):
                # trigger garbage collection in case the registered object isn't referenced anymore
                gc.collect(2)

            if self.identifier in registry:
                raise RuntimeError('Pulse with name already exists', self.identifier)

            registry[self.identifier] = self

    @property
    def identifier(self) -> Optional[str]:
        """The (optional) identifier of this Serializable. Either a non-empty string or None."""
        return self.__identifier

    def get_serialization_data(self, serializer: Optional['Serializer']=None) -> Dict[str, Any]:
        """Returns all data relevant for serialization as a dictionary containing only base types.

        Implementation hint:
        In the old serialization routines, if the Serializable contains complex objects which are itself
        Serializables, a serialized representation for these MUST be obtained by calling the dictify()
        method of serializer. The reason is that serializer may decide to either return a dictionary
        to embed or only a reference to the Serializable subelement. This is DEPRECATED behavior as of May 2018.
        In the new routines, this will happen automatically and every Serializable is only responsible for
        returning it's own data and leave nested Serializables in object form.

        For the transition time where both implementations are
        available, implementations of this method should support the old and new routines, using
        the presence of the serializer argument to differentiate between both. Don't make use of
        the implementation in this base class when implementing this method for the old routines.

        Args:
            serializer (Serializer): DEPRECATED (May 2018).A Serializer instance used to serialize
                complex subelements of this Serializable.
        Returns:
            A dictionary of Python base types (strings, integers, lists/tuples containing these,
                etc..) which fully represent the relevant properties of this Serializable for
                storing and later reconstruction as a Python object.
        """
        if serializer:
            warnings.warn("{c}.get_serialization_data(*) was called with a serializer argument, indicating deprecated behavior. Please switch to the new serialization routines.".format(c=self.__class__.__name__), DeprecationWarning)

        if self.identifier:
            return {self.type_identifier_name: self.get_type_identifier(), self.identifier_name: self.identifier}
        else:
            return {self.type_identifier_name: self.get_type_identifier()}

    @classmethod
    def get_type_identifier(cls) -> str:
        return "{}.{}".format(cls.__module__, cls.__name__)

    @classmethod
    def deserialize(cls, serializer: Optional['Serializer']=None, **kwargs) -> 'Serializable':
        """Reconstructs the Serializable object from a dictionary.

        Implementation hint:
        For greater clarity, implementations of this method should be precise in their return value,
        i.e., give their exact class name, and also replace the kwargs argument by a list of
        arguments required, i.e., those returned by get_serialization_data.
        Using old serialization routines, if this Serializable contains complex objects which are itself
        of type Serializable, their dictionary representations MUST be converted into objects using
        serializers deserialize() method. This is DEPRECATED behavior.
        Using the new routines, a serializable is only responsible to decode it's own dictionary,
        not those of nested objects (i.e., all incoming arguments are already processed by the
        serialization routines).
        For the transition time where both variants are
        available, implementations of this method should support the old and new routines, using
        the presence of the serializer argument to differentiate between both. For the new routines,
        just call this base class function.
        After the transition period, subclasses likely need not implement deserialize separately anymore at all.

        Args:
            serializer: DEPRECATED (May 2018). A serializer instance used when deserializing subelements.
            **kwargs: All relevant properties of the object as keyword arguments. For every (key,value)
                pair returned by get_serialization_data, the same pair is given as keyword argument as input
                to this method.
         """
        if serializer:
            warnings.warn("{c}.deserialize(*) was called with a serializer argument, indicating deprecated behavior. Please switch to the new serialization routines.".format(c=cls.__name__), DeprecationWarning)

        return cls(**kwargs)

    def renamed(self, new_identifier: str, registry: Optional[PulseRegistryType]=None) -> 'Serializable':
        """Returns a copy of the Serializable with its identifier set to new_identifier.

        Args:
            new_identifier: The identifier of the new copy of this Serializable.
            registry: The pulse registry the copy of this Serializable will register in. If None, the default pulse
                registry will be used. Optional.
        """
        data = self.get_serialization_data()
        data.pop(Serializable.type_identifier_name)
        data.pop(Serializable.identifier_name, None)
        return self.deserialize(registry=registry, identifier=new_identifier, **data)


class AnonymousSerializable:
    """Any object that can be converted into a serialized representation for storage and back which NEVER has an
    identifier. This class is used for implicit serialization and does not work necessarily with dicts.

    The type information is not saved explicitly but implicitly by the position in the JSON-document.

    See also:
        Serializable

    # todo (lumip, 2018-05-30): this does not really have a purpose, especially in the new serialization ecosystem.. we should deprecate and remove it
    """

    def get_serialization_data(self) -> Any:
        """Return all data relevant for serialization as a JSON compatible type that is accepted as constructor argument

        Returns:
            A JSON compatible type that can be used to construct an equal object.
        """
        raise NotImplementedError()


class Serializer(object):
    """Serializes Serializable objects and stores them persistently.

    DEPRECATED as of May 2018. Serializer will be superseeded by the new serialization routines and
    PulseStorage class.

    Serializer provides methods to enable the conversion of Serializable objects (including nested
    Serializables) into (nested) dictionaries and serialized JSON-encodings of these and vice-versa.
    Additionally, it can also store these representations persistently using a StorageBackend
    instance.

    See also:
        Serializable
    """

    __FileEntry = NamedTuple("FileEntry", [('serialization', str), ('serializable', Serializable)])

    def __init__(self, storage_backend: StorageBackend) -> None:
        """Creates a Serializer.

        Args:
            storage_backend (StorageBackend): The StorageBackend all objects will be stored in.
        """
        self.__subpulses = dict() # type: Dict[str, Serializer.__FileEntry]
        self.__storage_backend = storage_backend

        warnings.warn("Serializer is deprecated. Please switch to the new serialization routines.", DeprecationWarning)

    def dictify(self, serializable: Serializable) -> Union[str, Dict[str, Any]]:
        """Converts a Serializable into a dictionary representation.

        The Serializable is converted by calling its get_serialization_data() method. If it contains
        nested Serializables, these are also converted into dictionarys (or references), yielding
        a single dictionary representation of the outermost Serializable where all nested
        Serializables are either completely embedded or referenced by identifier.

        Args:
            serializable (Serializabe): The Serializable object to convert.
        Returns:
            A serialization dictionary, i.e., a dictionary of Python base types (strings, integers,
                lists/tuples containing these, etc..) which fully represent the relevant properties
                of the given Serializable for storing and later reconstruction as a Python object.
                Nested Serializables are either embedded or referenced by identifier.
        Raises:
            Exception if an identifier is assigned twice to different Serializable objects
                encountered by this Serializer during the conversion.
        See also:
            Serializable.get_serialization_data
        """
        repr_ = serializable.get_serialization_data(serializer=self)
        repr_['type'] = self.get_type_identifier(serializable)
        identifier = serializable.identifier
        if identifier is None:
            return repr_
        else:
            if identifier in self.__subpulses:
                if self.__subpulses[identifier].serializable is not serializable:
                    raise Exception("Identifier '{}' assigned twice.".format(identifier))
            else:
                self.__subpulses[identifier] = Serializer.__FileEntry(repr_, serializable)
            return identifier

    def __collect_dictionaries(self, serializable: Serializable) -> Dict[str, Dict[str, Any]]:
        """Converts a Serializable into a collection of dictionary representations.

        The Serializable is converted by calling its get_serialization_data() method. If it contains
        nested Serializables, these are also converted into dictionarys (or references), yielding
        a dictionary representation of the outermost Serializable where all nested
        Serializables are either completely embedded or referenced by identifier as it is returned
        by dictify. If nested Serializables shall be stored separately, their dictionary
        representations are collected. Collection_dictionaries returns a dictionary of all
        serialization dictionaries where the keys are the identifiers of the Serializables.

        Args:
            serializable (Serializabe): The Serializable object to convert.
        Returns:
            A dictionary containing serialization dictionary for each separately stored Serializable
                nested in the given Serializable.
        See also:
            dictify
        """
        self.__subpulses = dict()
        repr_ = self.dictify(serializable)
        filedict = dict()
        for identifier in self.__subpulses:
            filedict[identifier] = self.__subpulses[identifier].serialization
        if isinstance(repr_, dict):
            filedict[''] = repr_
        return filedict

    @staticmethod
    def get_type_identifier(obj: Any) -> str:
        """Returns a unique type identifier for any object.

        Args:
            obj: The object for which to obtain a type identifier.
        Returns:
            The type identifier as a string.
        """
        return "{}.{}".format(obj.__module__, obj.__class__.__name__)

    def serialize(self, serializable: Serializable, overwrite=False) -> None:
        """Serializes and stores a Serializable.

        The given Serializable and all nested Serializables that are to be stored separately will be
        converted into a serial string representation by obtaining their dictionary representation,
        encoding them as a JSON-string and storing them in the StorageBackend.

        If no identifier is given for the Serializable, "main" will be used.

        If an identifier is already in use in the StorageBackend, associated data will be replaced.

        Args:
            serializable (Serializable): The Serializable to serialize and store
        """
        warnings.warn("Serializer is deprecated. Please switch to the new serialization routines.", DeprecationWarning)
        repr_ = self.__collect_dictionaries(serializable)
        for identifier in repr_:
            storage_identifier = identifier
            if identifier == '':
                storage_identifier = 'main'
            json_str = json.dumps(repr_[identifier], indent=4, sort_keys=True, cls=ExtendedJSONEncoder)
            self.__storage_backend.put(storage_identifier, json_str, overwrite)

    def deserialize(self, representation: Union[str, Dict[str, Any]]) -> Serializable:
        """Loads a stored Serializable object or converts a dictionary representation back to the
            corresponding Serializable.

        Args:
            representation: A serialization dictionary representing a Serializable object or the
                identifier of a Serializable object to load from the StorageBackend.
        Returns:
            The Serializable object instantiated from its serialized representation.
        See also:
            Serializable.deserialize
        """
        warnings.warn("Serializer is deprecated. Please switch to the new serialization routines.", DeprecationWarning)
        if isinstance(representation, str):
            if representation in self.__subpulses:
                return self.__subpulses[representation].serializable

        if isinstance(representation, str):
            repr_ = json.loads(self.__storage_backend.get(representation))
            repr_['identifier'] = representation
        else:
            repr_ = dict(representation)

        module_name, class_name = repr_['type'].rsplit('.', 1)
        module = __import__(module_name, fromlist=[class_name])
        class_ = getattr(module, class_name)

        repr_to_store = repr_.copy()
        repr_.pop('type')

        serializable = class_.deserialize(self, **repr_)

        if 'identifier' in repr_:
            identifier = repr_['identifier']
            self.__subpulses[identifier] = self.__FileEntry(repr_, serializable)
        return serializable


class PulseStorage(MutableMapping[str, Serializable]):
    """The central storage management for pulses.

    Provides a dictionary interface for loading and storing pulses based on any StorageBackend implementation.
    Takes care of serialization and deserialization of pulses/Serializables in the process. Every Serializable
    with an identifier will be stored as a separate entity, even if it is nested in another Serializable that is stored.
    Serializables containing named Serializables will just store a reference to those which will be transparently
    resolved by PulseStorage during loading.

    PulseStorage employs caching, i.e., once a Serializable/pulse is loaded, it will be kept in memory and subsequent
    fetches will be instantaneous. At the same time, all changes to Serializables known to PulseStorage will immediately
    be flushed to the storage backend.

    Note that it is currently not possible to store a Serializable under a different identifier than the one it holds,
    i.e. you cannot store a Serializable `serializable` with identifier 'foo' under identifier 'bar' by calling
    `pulse_storage['bar'] = serializable`. This will currently result in a ValueError.

    It is also not possible to overwrite a Serializable using the dictionary interface. To explicitly overwrite a
    Serializable in the storage, use the `overwrite` method.

    PulseStorage can be used as the default pulse registry.
    All Serializables (and thus pulses) will automatically register themselves with the default pulse registry on
    construction unless an explicit other registry is provided to them as construction argument.
    This is intended to prevent accidental duplicate usage of identifiers by failing early. Setting
    a PulseStorage as pulse default registry also implies that all created Serializables are automatically stored
    in the storage backend.
    See Also:
        PulseStorage.set_to_default_registry
        PulseStorage.as_default_registry
    """
    StorageEntry = NamedTuple('StorageEntry', [('serialization', str), ('serializable', Serializable)])

    def __init__(self,
                 storage_backend: StorageBackend) -> None:
        """Create a PulseStorage instance.

        Args:
            storage_backend: The StorageBackend representing the permanent storage of the PulseStorage. Serializables
                are stored to and read from here.
        """
        self._storage_backend = storage_backend

        self._temporary_storage = dict() # type: Dict[str, StorageEntry]
        self._transaction_storage = None

    def _deserialize(self, serialization: str) -> Serializable:
        decoder = JSONSerializableDecoder(storage=self)
        serializable = decoder.decode(serialization)
        return serializable

    def _load_and_deserialize(self, identifier: str) -> StorageEntry:
        serialization = self._storage_backend[identifier]
        serializable = self._deserialize(serialization)
        self._temporary_storage[identifier] = PulseStorage.StorageEntry(serialization=serialization,
                                                                        serializable=serializable)
        return self._temporary_storage[identifier]

    @property
    def temporary_storage(self) -> Dict[str, StorageEntry]:
        """The in-memory temporary storage.

        Contains all Serializables that have been loaded during the lifetime of this PulseStorage object."""
        return self._temporary_storage

    def __contains__(self, identifier) -> bool:
        return identifier in self._temporary_storage or identifier in self._storage_backend

    def __getitem__(self, identifier: str) -> Serializable:
        """Fetch a Serializable.

        If the Serializable is not present in temporary storage, it will be loaded and deserialized from the storage
        backend.

        Args:
            identifier: The identifier of the Serializable to load.
        """
        if identifier not in self._temporary_storage:
            self._load_and_deserialize(identifier)
        return self._temporary_storage[identifier].serializable

    def __setitem__(self, identifier: str, serializable: Serializable) -> None:
        """Store a Serializable in the PulseStorage.

        Note that it is currently not possible to store a Serializable under a different identifier than the one it holds,
        i.e. you cannot store a Serializable `serializable` with identifier 'foo' under identifier 'bar' by calling
        `pulse_storage['bar'] = serializable`. This will currently result in a ValueError.

        It is also not possible to overwrite a Serializable using the dictionary interface. To explicitly overwrite a
        Serializable in the storage, use the `overwrite` method.

        Args:
            identifier: The identifier to store the Serializable under. Has to be identical to `serialziable.identifier`.
            serializable: The Serializable object to be stored.
        Raises:
            ValueError: if the given identifier argument does not match the identifier of the serializable
        """
        if identifier != serializable.identifier: # address issue #272: https://github.com/qutech/qupulse/issues/272
            raise ValueError("Storing a Serializable under a different than its own internal identifier is currently"
                             " not supported! If you want to rename the serializable, please use the "
                             "Serializable.renamed() method to obtain a renamed copy which can then be stored with "
                             "the new identifier.\n"
                             "If you think that storing under a different identifier without explicit renaming should"
                             "a supported feature, please contribute to our ongoing discussion about this on:\n"
                             "https://github.com/qutech/qupulse/issues/272")
        if identifier in self._temporary_storage:
            if self.temporary_storage[identifier].serializable is serializable:
                return
            else:
                raise RuntimeError('Identifier assigned twice with different objects', identifier)
        elif identifier in self._storage_backend:
            raise RuntimeError('Identifier already assigned in storage backend', identifier)
        self.overwrite(identifier, serializable)

    def __delitem__(self, identifier: str) -> None:
        """Delete an item from temporary storage and storage backend.

        Does not raise an error if the deleted pulse is only in the storage backend. Assumes that all pulses
        contained in temporary storage are always also contained in the storage backend.
        Args:
            identifier: Identifier of the Serializable to delete
        """
        del self._storage_backend[identifier]
        try:
            del self._temporary_storage[identifier]
        except KeyError:
            pass

    @property
    def contents(self) -> Iterable[str]:
        return self._storage_backend.list_contents()

    def __len__(self) -> int:
        return len(self._storage_backend)

    def __iter__(self) -> Iterator[str]:
        return iter(self._storage_backend)

    def overwrite(self, identifier: str, serializable: Serializable) -> None:
        """Explicitly overwrites a pulse.

        Calling this method will overwrite the entity currently stored under the given identifier by the
        provided serializable. It does _not_ overwrite nested Serializable objects contained in serializable. If you
        want to overwrite those as well, do that explicitely.

        Args:
              identifier: The identifier to store serializable under.
            serializable: The Serializable object to be stored.
        """

        is_transaction_begin = (self._transaction_storage is None)
        try:
            if is_transaction_begin:
                self._transaction_storage = dict()

            encoder = JSONSerializableEncoder(self, sort_keys=True, indent=4)

            serialization_data = serializable.get_serialization_data()
            serialized = encoder.encode(serialization_data)
            self._transaction_storage[identifier] = self.StorageEntry(serialized, serializable)

            if is_transaction_begin:
                for identifier, entry in self._transaction_storage.items():
                    self._storage_backend.put(identifier, entry.serialization, overwrite=True)
                self._temporary_storage.update(**self._transaction_storage)

        finally:
            if is_transaction_begin:
                self._transaction_storage = None

    def clear(self) -> None:
        """Clears the temporary storage.

        Does not affect the storage backend."""
        self._temporary_storage.clear()

    @contextmanager
    def as_default_registry(self) -> Any:
        """Returns context manager to use this PulseStorage as the default pulse registry only within a with-statement."""
        global default_pulse_registry
        previous_registry = default_pulse_registry
        default_pulse_registry = self
        try:
            yield self
        finally:
            default_pulse_registry = previous_registry

    def set_to_default_registry(self) -> None:
        """Promotes this PulseStorage object to be the default pulse registry.

        All Serializables (and thus pulses) will automatically register themselves with the default pulse registry on
        construction unless an explicit other registry is provided to them as construction argument.
        This is intended to prevent accidental duplicate usage of identifiers by failing early. Setting
        a PulseStorage as pulse default registry also implies that all created Serializables are automatically stored
        in the storage backend."""
        global default_pulse_registry
        default_pulse_registry = self


class JSONSerializableDecoder(json.JSONDecoder):
    """JSONDecoder for Serializables.

    Automatically follows references to nested Serializables during deserializing."""

    def __init__(self, storage: Mapping, *args, **kwargs) -> None:
        """Creates a new JSONSerialzableDecoder object.

        Args:
            storage: Any mapping of identifier to Serializable objects. Will be used to resolve references to nested
                Serializables. Usually a PulseStorage object.
            *args: Any other positional argument will be passed on to JSONDecoder constructor.
            **kwargs: Any keyword argument will be passed on to JSONDecoder.
        See Also:
            JSONDecoder
        """
        super().__init__(*args, object_hook=self.filter_serializables, **kwargs)

        self.storage = storage

    def filter_serializables(self, obj_dict) -> Any:
        if Serializable.type_identifier_name in obj_dict:
            type_identifier = obj_dict.pop(Serializable.type_identifier_name)

            if Serializable.identifier_name in obj_dict:
                obj_identifier = obj_dict.pop(Serializable.identifier_name)
            else:
                obj_identifier = None

            if type_identifier == 'reference':
                if not obj_identifier:
                    raise RuntimeError('Reference without identifier')
                return self.storage[obj_identifier]

            else:
                deserialization_callback = SerializableMeta.deserialization_callbacks[type_identifier]

                # if the storage is the default registry, we would get conflicts when the Serializable tries to register
                # itself on construction. Pass an empty dict as registry keyword argument in this case.
                # calling PulseStorage objects will take care of registering.
                # (solution to issue #301: https://github.com/qutech/qupulse/issues/301 )
                registry = None
                if get_default_pulse_registry() is self.storage:
                    registry = dict()

                return deserialization_callback(identifier=obj_identifier, registry=registry, **obj_dict)
        return obj_dict


class JSONSerializableEncoder(json.JSONEncoder):
    """JSONEncoder for Serializables.

    Ensures that nested Serializables are stored as separate entities and embedded in the parent Serializable's
    serialization by reference."""

    def __init__(self, storage: MutableMapping, *args, **kwargs) -> None:
        """Creates a new JSONSerialzableDecoder object.

            Args:
                storage: Any mapping of identifier to Serializable objects. Will be used to store nested
                    Serializables. Usually a PulseStorage object.
                *args: Any other positional argument will be passed on to JSONEncoder constructor.
                **kwargs: Any keyword argument will be passed on to JSONEncoder.
            See Also:
                JSONEncoder
        """
        super().__init__(*args, **kwargs)

        self.storage = storage

    def default(self, o: Any) -> Any:
        if isinstance(o, Serializable):
            if o.identifier:
                if o.identifier not in self.storage:
                    self.storage[o.identifier] = o
                elif o is not self.storage[o.identifier]:
                    raise RuntimeError('Trying to store a subpulse with an identifier that is already taken.')


                return {Serializable.type_identifier_name: 'reference',
                        Serializable.identifier_name: o.identifier}
            else:
                return o.get_serialization_data()

        elif isinstance(o, AnonymousSerializable):
            return o.get_serialization_data()

        elif type(o) is set:
            return list(o)

        else:
            return super().default(o)


class ExtendedJSONEncoder(json.JSONEncoder):
    """Encodes AnonymousSerializable and sets as lists.

    Used by Serializer.

    Deprecated as of May 2018. To be replaced by JSONSerializableEncoder."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def default(self, o: Any) -> Any:
        if isinstance(o, AnonymousSerializable):
            return o.get_serialization_data()
        elif type(o) is set:
            return list(o)
        else:
            return super().default(o)


def convert_stored_pulse_in_storage(identifier: str, source_storage: StorageBackend, dest_storage: StorageBackend) -> None:
    """Converts a pulse from the old to the new serialization format.

    The pulse with the given identifier is completely (including subpulses) converted from the old serialization format
    read from a given source storage to the new serialization format and written to a given destination storage.

    Args:
        identifier (str): The identifier of the pulse to convert.
        source_storage (StorageBackend): A StorageBackend containing the pulse identified by the identifier argument in the old serialization format.
        dest_storage (StorageBackend): A StorageBackend the converted pulse will be written to in the new serialization format.
    Raises:
        ValueError: if the dest_storage StorageBackend contains identifiers also assigned in source_storage.
    """
    if dest_storage.list_contents().intersection(source_storage.list_contents()):
        raise ValueError("dest_storage already contains pulses with the same ids. Aborting to prevent inconsistencies for duplicate keys.")
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        serializer = Serializer(source_storage)
        pulse_storage = PulseStorage(dest_storage)
        serializable = serializer.deserialize(identifier)
        pulse_storage.overwrite(identifier, serializable)


def convert_pulses_in_storage(source_storage: StorageBackend, dest_storage: StorageBackend) -> None:
    """Converts all pulses from the old to the new serialization format.

        All pulses in a given source storage are completely (including subpulses) converted from the old serialization format
        to the new serialization format and written to a given destination storage.

        Args:
            source_storage (StorageBackend): A StorageBackend containing pulses in the old serialization format.
            dest_storage (StorageBackend): A StorageBackend the converted pulses will be written to in the new serialization format.
        Raises:
            ValueError: if the dest_storage StorageBackend contains identifiers also assigned in source_storage.
        """
    if dest_storage.list_contents().intersection(source_storage.list_contents()):
        raise ValueError("dest_storage already contains pulses with the same ids. Aborting to prevent inconsistencies for duplicate keys.")
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        serializer = Serializer(source_storage)
        pulse_storage = PulseStorage(dest_storage)
        for identifier in source_storage.list_contents():
            serializable = serializer.deserialize(identifier)
            pulse_storage.overwrite(identifier, serializable)
