from abc import ABCMeta

class Serializable(meta = ABCMeta):
    
    @property
    def serialization_data(self) -> None:
        return self.__dict__

class Serializer(object):
    def __init__(self) -> None:
        pass
    
    def serialize(self, serializable: Serializable) -> None:
        pass