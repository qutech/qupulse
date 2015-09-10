from abc import ABCMeta
from typing import Dict, Any, Optional, NamedTuple
import json


class Serializable(metaclass = ABCMeta):

    def __init__(self, identifier: Optional[str]) -> None:
        super().__init__()
        self.__identifier = identifier

    @property
    def identifier(self) -> Optional[str]:
        return self.__identifier

    def get_serialization_data(self, serializer: 'Serializer') -> Dict[str, Any]:
        return self.__dict__


class Serializer(object):

    FileEntry = NamedTuple("FileEntry", [('serialization', str), ('serializable', Serializable)])

    def __init__(self) -> None:
        self.__files = dict() # type: Dict[str, FileEntry]
        pass
    
    def serialize(self, serializable: Serializable) -> str:
        serialized = json.dumps(serializable.get_serialization_data(self))
        identifier = serializable.identifier
        if identifier is None:
            return serialized
        else:
            if identifier in self.__files.keys():
                if self.__files[identifier].serializable is not serializable:
                    raise Exception("Identifier '{}' assigned twice.".format(identifier))
            else:
                self.__files[identifier] = serialized
            return identifier
