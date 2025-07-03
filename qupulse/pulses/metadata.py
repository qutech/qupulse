import collections
import dataclasses

from typing import Literal, Optional


SingleWaveformStrategy = Literal['always']


@dataclasses.dataclass(frozen=False, eq=False, repr=False)
class TemplateMetadata:
    """This class is used to store metadata for pulse templates.

    It is the only volatile part of pulse templates and thus does not participate in it's equality operation.
    To enforce that this class does not implement the equality operator.

    It implements the serializable protocol.
    """

    to_single_waveform: Optional[SingleWaveformStrategy] = dataclasses.field(default=None)

    def __init__(self, to_single_waveform: Optional[SingleWaveformStrategy] = None, **kwargs):
        # TODO: generate this init automatically
        #       The reason for the custom init is that we want to allow additional kwargs
        self.to_single_waveform = to_single_waveform
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self):
        args = ",".join(f"{name}={value!r}"
                        for name, value in self.get_serialization_data().items())
        return f'{self.__class__.__name__}({args})'

    def get_serialization_data(self):
        data = vars(self)

        for field in dataclasses.fields(self):
            if field.default is not dataclasses.MISSING:
                if data[field.name] == field.default:
                    del data[field.name]

        return data

    def __bool__(self):
        return bool(self.get_serialization_data())
