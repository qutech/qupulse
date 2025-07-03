import collections
import dataclasses

from typing import Literal, Optional


SingleWaveformStrategy = Literal['always']


@dataclasses.dataclass(frozen=False, eq=False, repr=False)
class TemplateMetadata:
    to_single_waveform: Optional[SingleWaveformStrategy] = dataclasses.field(default=None)

    def __repr__(self):
        args = ",".join(f"{name}={value!r}" for name, value in self.get_serialization_data())
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
