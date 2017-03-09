"""Import dummy modules if actual modules not installed. Sets dummy modules in sys so subsequent imports
use the dummies"""

import sys
from typing import Set

class dummy_package:
    pass

class dummy_pytabor(dummy_package):
    pass

class dummy_pyvisa(dummy_package):
    pass

class dummy_teawg(dummy_package):
    model_properties_dict = dict()
    class TEWXAwg:
        def __init__(self, *args, **kwargs):
            pass
        send_cmd = __init__
        send_query = send_cmd
        select_channel = send_cmd
        send_binary_data = send_cmd
        download_sequencer_table = send_cmd

class dummy_atsaverage(dummy_package):
    class atsaverage(dummy_package):
        pass
    class alazar(dummy_package):
        pass
    class core(dummy_package):
        class AlazarCard:
            model = 'DUMMY'
            minimum_record_size = 256
    class config(dummy_package):
        class CaptureClockConfig:
            def numeric_sample_rate(self, card):
                return 10**8
        class ScanlineConfiguration:
            pass
        ScanlineConfiguration.captureClockConfiguration = CaptureClockConfig()
    class operations(dummy_package):
        class OperationDefinition:
            pass
    class masks(dummy_package):
        class Mask:
            pass
        class CrossBufferMask:
            pass


def import_package(name, package) -> Set[dummy_package]:
    imported = set()
    sys.modules[name] = package
    imported.add(package)
    for attr in dir(package):
        if isinstance(getattr(package, attr), type) and issubclass(getattr(package, attr), dummy_package):
            imported |= import_package(name + '.' + attr, getattr(package, attr))
    return imported


failed_imports = set()
try:
    import pytabor
except ImportError:
    failed_imports |= import_package('pytabor', dummy_pytabor)

try:
    import pyvisa
except ImportError:
    failed_imports |= import_package('pyvisa', dummy_pyvisa)

try:
    import teawg
except ImportError:
    failed_imports |= import_package('teawg', dummy_teawg)

try:
    import atsaverage
    import atsaverage.config
except ImportError:
    failed_imports |= import_package('atsaverage', dummy_atsaverage)

