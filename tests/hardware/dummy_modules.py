"""Import dummy modules if actual modules not installed. Sets dummy modules in sys so subsequent imports
use the dummies"""

import sys
from typing import Set
import unittest.mock

class dummy_package:
    pass


class dummy_atsaverage(dummy_package):
    class atsaverage(dummy_package):
        pass
    class alazar(dummy_package):
        pass
    class core(dummy_package):
        class AlazarCard:
            model = 'DUMMY'
            minimum_record_size = 256
            def __init__(self):
                self._startAcquisition_calls = []
                self._applyConfiguration_calls = []
            def startAcquisition(self, x: int):
                self._startAcquisition_calls.append(x)
            def applyConfiguration(self, config):
                self._applyConfiguration_calls.append(config)
    class config(dummy_package):
        class CaptureClockConfig:
            def numeric_sample_rate(self, card):
                return 10**8
        class ScanlineConfiguration:
            def __init__(self):
                self._apply_calls = []
            def apply(self, card, print_debug_output):
                self._apply_calls.append((card, print_debug_output))
            aimedBufferSize = unittest.mock.PropertyMock(return_value=2**22)
        ScanlineConfiguration.captureClockConfiguration = CaptureClockConfig()
    class operations(dummy_package):
        class OperationDefinition:
            pass
    class masks(dummy_package):
        class Mask:
            pass
        class CrossBufferMask:
            pass


def import_package(name, package=None) -> Set[dummy_package]:
    if package is None:
        package_dict = dict(atsaverage=dummy_atsaverage)
        if name in package_dict:
            package = package_dict[name]
        else:
            raise KeyError('Unknown package', name)

    imported = set()
    sys.modules[name] = package
    imported.add(package)
    for attr in dir(package):
        if isinstance(getattr(package, attr), type) and issubclass(getattr(package, attr), dummy_package):
            imported |= import_package(name + '.' + attr, getattr(package, attr))
    return imported


def replace_missing():
    failed_imports = set()

    try:
        import atsaverage
        import atsaverage.config
    except ImportError:
        failed_imports |= import_package('atsaverage', dummy_atsaverage)
    return failed_imports

