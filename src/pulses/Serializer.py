from abc import ABCMeta, abstractmethod, abstractstaticmethod
from typing import Dict, Any, Optional, NamedTuple, Union
import os.path
import json


class StorageBackend(metaclass = ABCMeta):

    @abstractmethod
    def put(self, identifier: str, data: str, overwrite: bool=False) -> None:
        """Store the data string identified by identifier."""

    @abstractmethod
    def get(self, identifier: str) -> str:
        """Retrieve the data string with the given identifier."""

    @abstractmethod
    def exists(self, identifier: str) -> bool:
        """Return True, if data is stored for the given identifier."""


class FilesystemBackend(StorageBackend):

    def __init__(self, root: str='.') -> None:
        if not os.path.isdir(root):
            raise NotADirectoryError()
        self.__root = os.path.abspath(root)

    def put(self, identifier: str, data: str, overwrite: bool=False) -> None:
        path = os.path.join(self.__root, identifier)
        if self.exists(identifier) and not overwrite:
            raise FileExistsError()
        with open(path, 'w') as f:
            f.write(data)

    def get(self, identifier: str) -> str:
        path = os.path.join(self.__root, identifier)
        with open(path) as f:
            return f.read()

    def exists(self, identifier: str) -> bool:
        path = os.path.join(self.__root, identifier)
        return os.path.isfile(path)


class CachingBackend(StorageBackend):

    def __init__(self, backend: StorageBackend) -> None:
        self.__backend = backend
        self.__cache = {}

    def put(self, identifier: str, data: str, overwrite: bool=False) -> None:
        if identifier in self.__cache and not overwrite:
            raise FileExistsError()
        self.__backend.put(identifier, data, overwrite)
        self.__cache[identifier] = data

    def get(self, identifier: str) -> str:
        if identifier not in self.__cache:
            self.__cache[identifier] = self.__backend.get(identifier)
        return self.__cache[identifier]

    def exists(self, identifier: str) -> bool:
        return self.__backend.exists(identifier)


class Serializable(metaclass = ABCMeta):

    def __init__(self, identifier: Optional[str] = None) -> None:
        super().__init__()
        if identifier == '':
            raise ValueError("Identifier must not be empty.")
        self.__identifier = identifier

    @property
    def identifier(self) -> Optional[str]:
        return self.__identifier

    @abstractmethod
    def get_serialization_data(self, serializer: 'Serializer') -> Dict[str, Any]:
        """Return all data relevant for serialization as a dictionary containing only base types."""

    @abstractstaticmethod
    def deserialize(serializer: 'Serializer', **kwargs) -> 'Serializable':
        """Reconstruct the Serializable object from a dictionary containing all relevant information as obtained from get_serialization_data."""


class Serializer(object):

    FileEntry = NamedTuple("FileEntry", [('serialization', str), ('serializable', Serializable)])

    def __init__(self, storage_backend: StorageBackend) -> None:
        self.__subpulses = dict() # type: Dict[str, Serializer.FileEntry]
        self.__storage_backend = storage_backend

    def _serialize_subpulse(self, serializable: Serializable) -> Union[str, Dict[str, Any]]:
        repr_ = serializable.get_serialization_data(self)
        identifier = serializable.identifier
        if identifier is None:
            return repr_
        else:
            if identifier in self.__subpulses:
                if self.__subpulses[identifier].serializable is not serializable:
                    raise Exception("Identifier '{}' assigned twice.".format(identifier))
            else:
                self.__subpulses[identifier] = Serializer.FileEntry(repr_, serializable)
            return identifier
    
    def dictify(self, serializable: Serializable) -> Dict[str, Dict[str, Any]]:
        self.__subpulses = dict()
        repr_ = self._serialize_subpulse(serializable)
        filedict = dict()
        for identifier in self.__subpulses:
            filedict[identifier] = self.__subpulses[identifier].serialization
        if isinstance(repr_, dict):
            filedict[''] = repr_
        return filedict

    @staticmethod
    def get_type_identifier(obj: Any) -> str:
        return "{}.{}".format(obj.__module__, obj.__class__.__name__)

    def serialize(self, serializable: Serializable) -> None:
        repr_ = self.dictify(serializable)
        for identifier in repr_:
            storage_identifier = identifier
            if identifier == '':
                storage_identifier = 'main'
            self.__storage_backend.put(storage_identifier, json.dumps(repr_[identifier], indent=4, sort_keys=True))

    def deserialize(self, representation: Union[str, Dict[str, Any]]) -> Serializable:
        if isinstance(representation, str):
            repr_ = json.loads(self.__storage_backend.get(representation))
            repr_['identifier'] = representation
        else:
            repr_ = dict(representation)

        module_name, class_name = repr_['type'].rsplit('.', 1)
        module = __import__(module_name, fromlist=[class_name])
        class_ = getattr(module, class_name)

        repr_.pop('type')
        return class_.deserialize(self, **repr_)
