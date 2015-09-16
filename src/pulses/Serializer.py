from abc import ABCMeta
from typing import Dict, Any, Optional, NamedTuple, Union
import json


class Serializable(metaclass = ABCMeta):

    def __init__(self, identifier: Optional[str]) -> None:
        super().__init__()
        self.__identifier = identifier

    @property
    def identifier(self) -> Optional[str]:
        if self.__identifier == '':
            raise ValueError("Identifier must not be empty.")
        return self.__identifier

    def get_serialization_data(self, serializer: 'Serializer') -> Dict[str, Any]:
        return self.__dict__


class Serializer(object):

    FileEntry = NamedTuple("FileEntry", [('serialization', str), ('serializable', Serializable)])

    def __init__(self) -> None:
        self.__subpulses = dict() # type: Dict[str, FileEntry]
        pass

    def _serialize_subpulse(self, serializable: Serializable) -> Union[str, Dict[str, Any]]:
        repr_ = serializable.get_serialization_data(self)
        repr_['type'] = str(serializable.__class__)
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
    
    def serialize(self, serializable: Serializable) -> Dict[str, Dict[str, Any]]:
        self.__subpulses = dict()
        repr_ = self._serialize_subpulse(serializable)
        filedict = dict()
        for identifier in self.__subpulses:
            filedict[identifier] = self.__subpulses[identifier].serialization
        if isinstance(repr_, dict):
            filedict[''] = repr_
        return filedict

