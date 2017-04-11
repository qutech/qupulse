"""Import dummy modules if actual modules not installed. Sets dummy modules in sys so subsequent imports
use the dummies"""

import sys
from typing import Set

class dummy_package:
    pass

class dummy_pytabor(dummy_package):
    pass

class dummy_pyvisa(dummy_package):
    class resources(dummy_package):
        class messagebased(dummy_package):
            class MessageBasedResource:
                def __init__(self, *args, **kwargs):
                    self.logged_writes = []
                    self.logged_asks = []
                def write(self, *args, **kwargs):
                    self.logged_writes.append((args, kwargs))
                def ask(self, *args, **kwargs):
                    self.logged_asks.append((args, kwargs))
                    return ';'.join( '0, bla'*args[0].count('?') )
dummy_pyvisa.resources.MessageBasedResource = dummy_pyvisa.resources.messagebased.MessageBasedResource


class dummy_teawg(dummy_package):
    model_properties_dict = dict()
    class TEWXAwg:
        def __init__(self, *args, **kwargs):
            self.logged_commands = []
            self.logged_queries = []
            self._visa_inst = dummy_pyvisa.resources.MessageBasedResource()
        @property
        def visa_inst(self):
            return self._visa_inst
        def send_cmd(self, *args, **kwargs):
            self.logged_commands.append((args, kwargs))
        def send_query(self, *args, **kwargs):
            self.logged_queries.append((args, kwargs))
            return 0
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

