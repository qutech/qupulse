from collections import Callable

import typing

from qupulse.hardware.awgs.base import AWGFeature


class DeviceMirrorFeature(AWGFeature):
    def __init__(self, main_instrument: Callable, mirrored_instruments: Callable,
                 all_devices: Callable):
        self._main_instrument = main_instrument
        self._mirrored_instruments = mirrored_instruments
        self._all_devices = all_devices

    def main_instrument(self) -> object:
        return self.main_instrument()

    def mirrored_instruments(self) -> typing.Any:
        return self.mirrored_instruments()

    def all_devices(self) -> typing.Any:
        return self.all_devices()
