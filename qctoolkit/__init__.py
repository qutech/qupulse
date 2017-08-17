import typing

__all__ = ["MeasurementWindow", "ChannelID"]

MeasurementWindow = typing.Tuple[str, float, float]
ChannelID = typing.Union[str, int]
