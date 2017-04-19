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
                    self.answers = dict()
                    self.default_answer = '0, bla'

                def write(self, *args, **kwargs):
                    self.logged_writes.append((args, kwargs))

                def ask(self, *args, **kwargs):
                    self.logged_asks.append((args, kwargs))
                    ques = args[0].split(';')
                    ques = [q.strip(' ?') for q in ques if q.strip().endswith('?')]
                    answers = [self.answers[q] if q in self.answers else self.default_answer
                               for q in ques]
                    return ';'.join(answers)
dummy_pyvisa.resources.MessageBasedResource = dummy_pyvisa.resources.messagebased.MessageBasedResource


class dummy_teawg(dummy_package):
    model_properties_dict = {
        'model_name': 'Dummy_WX2184',  # the model name
        'num_parts': 2,  # number of instrument parts
        'chan_per_part': 2,  # number of channels per part
        'seg_quantum': 16,  # segment-length quantum
        'min_seg_len': 192,  # minimal segment length
        'max_arb_mem': 32E6,  # maximal arbitrary-memory (points per channel)
        'min_dac_val': 0,  # minimal DAC value
        'max_dac_val': 2 ** 14 - 1,  # maximal DAC value
        'max_num_segs': 32E+3,  # maximal number of segments
        'max_seq_len': 48 * 1024 - 2,  # maximal sequencer-table length (# rows)
        'min_seq_len': 3,  # minimal sequencer-table length (# rows)
        'max_num_seq': 1000,  # maximal number of sequencer-table
        'max_aseq_len': 48 * 1024 - 2,  # maximal advanced-sequencer table length
        'min_aseq_len': 3,  # minimal advanced-sequencer table length
        'min_sclk': 75e6,  # minimal sampling-rate (samples/seconds)
        'max_sclk': 2300e6,  # maximal sampling-rate (samples/seconds)
        'digital_support': False,  # is digital-wave supported?
    }
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
            return self._visa_inst.ask(*args, **kwargs)
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


def import_package(name, package=None) -> Set[dummy_package]:
    if package is None:
        package_dict = dict(atsaverage=dummy_atsaverage,
                            pyvisa=dummy_pyvisa,
                            pytabor=dummy_pytabor,
                            teawg=dummy_teawg)
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
    return failed_imports

