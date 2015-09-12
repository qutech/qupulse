# encoding: utf-8
from abc import ABCMeta, abstractmethod, abstractproperty
from typing import Dict, Any
import json
import os.path
import ipdb

class Serializer(json.JSONEncoder):
    def default(self, obj):
        # Handle sets and other iterators:
        try:
            data = iter(obj)
        except TypeError:
            pass
        else:
            return list(obj)
        # Handle Serializable objects
        try:
            # this might be the wrong place (or get_serialization_data is in the wrong place)
            if obj.identifier:
                data = dict(type='JSON', name=obj.identifier)
            else:
                data = obj.get_serialization_data()
        except TypeError:
            pass
        else:
            return data
        # everything failed, raise TypeError
        super().default(obj)

    def serialize(self, obj: 'Serializable') -> str:
        '''Takes a serializable object and returns a JSON string representation'''
        default = lambda obj: self.default(obj)
        return json.dumps(obj, default = default)

class PulseTemplateJSONEncoder(json.JSONEncoder):
    def default(self, obj: 'Serializable'):
        try:
            res = super().default(obj)
        except TypeError:
            if isinstance(obj, set):
                data = list(obj)
            else:
                data = obj.get_serialization_data()
            res = super().default(data)
        return res

class StorageBackend(metaclass = ABCMeta):
    @abstractmethod
    def put(data: str, identifier: str):
        '''store the data string identified by identifier'''

    @abstractmethod
    def get(identifier: str) -> str:
        '''Retrieves the data string with the given identifier'''
        pass

class FilesystemBackend(StorageBackend):
    def __init__(self, root='.'):
        if not os.path.isdir(root):
            raise Exception
        self.__root = os.path.abspath(root)

    def put(self, data: str, identifier: str):
        path = os.path.join(self.__root, identifier)
        if os.path.isfile(path):
            raise Exception # file already exists
        with open(path, 'w') as f:
            f.write(data)

    def get(self, identifier):
        path = os.path.join(self.__root, identifier)
        with open(path) as f:
            return f.read()

class CachingBackend(StorageBackend):
    def __init__(self, backend):
        self.backend = backend
        self.cache = {}

    def put(self, data: str, identifier: str):
        self.cache[identifier] = data
        self.backend.put(data, identifier)

    def get(self, identifier):
        if identifier in self.cache.keys():
            return self.cache[identifier]
        else:
            self.backend.get(identifier)
    

class Serializable(metaclass = ABCMeta):
    @abstractproperty
    def identifier(self) -> str:
        '''Return the object's unique name, which is also the filename that it
        is serialized to.'''
        pass

    @abstractmethod
    def get_serialization_data(self) -> Dict[str, Any]:
        '''return an object representation in terms of built-in python types
        that can easily be serialized by a Serializer backend.'''
        pass

    # @abstractmethod # TODO: make virtual
    def from_serialization_data(self, data: Dict[str, Any]) -> 'Serializable':
        '''Reconstructs an object from a representation in built-in types'''
        pass

