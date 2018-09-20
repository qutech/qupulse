from typing import Union, Dict, Any, Callable, Iterator

from qupulse.serialization import Serializer, Serializable, StorageBackend


class DummyStorageBackend(StorageBackend):

    def __init__(self) -> None:
        self.stored_items = dict()
        self.times_get_called = 0
        self.times_put_called = 0
        self.times_exists_called = 0

    def get(self, identifier: str) -> str:
        self.times_get_called += 1
        if identifier not in self.stored_items:
            raise KeyError(identifier)
        return self.stored_items[identifier]

    def put(self, identifier: str, data: str, overwrite: bool=False) -> None:
        self.times_put_called += 1
        if identifier in self.stored_items and not overwrite:
            raise FileExistsError()
        self.stored_items[identifier] = data

    def exists(self, identifier: str) -> bool:
        self.times_exists_called += 1
        return identifier in self.stored_items

    def delete(self, identifier: str) -> None:
        del self.stored_items[identifier]

    def __iter__(self) -> Iterator[str]:
        return iter(self.stored_items)


class DummySerializer(Serializer):

    def __init__(self,
                 serialize_callback: Callable[[Serializable], str] = lambda x: "{}".format(id(x)),
                 identifier_callback: Callable[[Serializable], str] = lambda x: "{}".format(id(x)),
                 deserialize_callback: Callable[[Any], str] = lambda x: x) -> None:
        self.backend = DummyStorageBackend()
        self.serialize_callback = serialize_callback
        self.identifier_callback = identifier_callback
        self.deserialize_callback = deserialize_callback
        super().__init__(self.backend)
        self.subelements = dict()

    def dictify(self, serializable: Serializable) -> None:
        identifier = self.identifier_callback(serializable)
        self.subelements[identifier] = serializable
        return self.serialize_callback(serializable)

    def deserialize(self, representation: Union[str, Dict[str, Any]]) -> Serializable:
        return self.subelements[self.deserialize_callback(representation)]

