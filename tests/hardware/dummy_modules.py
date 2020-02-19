"""Import dummy modules if actual modules not installed. Sets dummy modules in sys so subsequent imports
use the dummies"""

import sys
from typing import Set
import unittest.mock

class dummy_package:
    pass

class dummy_pytabor(dummy_package):
    @staticmethod
    def open_session(*args, **kwargs):
        return None

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

                def query(self, *args, **kwargs):
                    self.logged_asks.append((args, kwargs))
                    ques = args[0].split(';')
                    ques = [q.strip(' ?') for q in ques if q.strip().endswith('?')]
                    answers = [self.answers[q] if q in self.answers else self.default_answer
                               for q in ques]
                    return ';'.join(answers)


dummy_pyvisa.resources.MessageBasedResource = dummy_pyvisa.resources.messagebased.MessageBasedResource


class dummy_teawg(dummy_package):
    # WX2184 Properties
    _wx2184_properties = {
        'model_name': 'WX2184',  # the model name
        'fw_ver': 0.0,  # the firmware version
        'serial_num': '0' * 9,  # serial number
        'num_parts': 2,  # number of instrument parts
        'chan_per_part': 2,  # number of channels per part
        'seg_quantum': 16,  # segment-length quantum
        'min_seg_len': 192,  # minimal segment length
        'max_arb_mem': 32E6,  # maximal arbitrary-memory (points per channel)
        'min_dac_val': 0,  # minimal DAC value
        'max_dac_val': 2 ** 14 - 1,  # maximal DAC value
        'max_num_segs': 32E+3,  # maximal number of segments
        'max_seq_len': 48 * 1024,  # maximal sequencer-table length (# rows)
        'min_seq_len': 3,  # minimal sequencer-table length (# rows)
        'max_num_seq': 1000,  # maximal number of sequencer-table
        'max_aseq_len': 48 * 1024 - 2,  # maximal advanced-sequencer table length
        'min_aseq_len': 3,  # minimal advanced-sequencer table length
        'min_sclk': 75e6,  # minimal sampling-rate (samples/seconds)
        'max_sclk': 2300e6,  # maximal sampling-rate (samples/seconds)
        'digital_support': False,  # is digital-wave supported?
    }

    # WX1284 Definitions
    _wx1284_properties = {
        'model_name': 'WX1284',  # the model name
        'fw_ver': 0.0,  # the firmware version
        'serial_num': '0' * 9,  # serial number
        'num_parts': 2,  # number of instrument parts
        'chan_per_part': 2,  # number of channels per part
        'seg_quantum': 16,  # segment-length quantum
        'min_seg_len': 192,  # minimal segment length
        'max_arb_mem': 32E6,  # maximal arbitrary-memory (points per channel)
        'min_dac_val': 0,  # minimal DAC value
        'max_dac_val': 2 ** 14 - 1,  # maximal DAC value
        'max_num_segs': 32E+3,  # maximal number of segments
        'max_seq_len': 48 * 1024,  # maximal sequencer-table length (# rows)
        'min_seq_len': 3,  # minimal sequencer-table length (# rows)
        'max_num_seq': 1000,  # maximal number of sequencer-table
        'max_aseq_len': 48 * 1024 - 2,  # maximal advanced-sequencer table length
        'min_aseq_len': 3,  # minimal advanced-sequencer table length
        'min_sclk': 75e6,  # minimal sampling-rate (samples/seconds)
        'max_sclk': 1250e6,  # maximal sampling-rate (samples/seconds)
        'digital_support': False,  # is digital-wave supported?
    }

    # WX2182C Definitions
    _wx2182C_properties = {
        'model_name': 'WX2182C',  # the model name
        'fw_ver': 0.0,  # the firmware version
        'serial_num': '0' * 9,  # serial number
        'num_parts': 2,  # number of instrument parts
        'chan_per_part': 1,  # number of channels per part
        'seg_quantum': 16,  # segment-length quantum
        'min_seg_len': 192,  # minimal segment length
        'max_arb_mem': 32E6,  # maximal arbitrary-memory (points per channel)
        'min_dac_val': 0,  # minimal DAC value
        'max_dac_val': 2 ** 14 - 1,  # maximal DAC value
        'max_num_segs': 32E+3,  # maximal number of segments
        'max_seq_len': 48 * 1024,  # maximal sequencer-table length (# rows)
        'min_seq_len': 3,  # minimal sequencer-table length (# rows)
        'max_num_seq': 1000,  # maximal number of sequencer-table
        'max_aseq_len': 1000,  # maximal advanced-sequencer table length
        'min_aseq_len': 3,  # minimal advanced-sequencer table length
        'min_sclk': 10e6,  # minimal sampling-rate (samples/seconds)
        'max_sclk': 2.3e9,  # maximal sampling-rate (samples/seconds)
        'digital_support': False,  # is digital-wave supported?
    }

    # WX1282C Definitions
    _wx1282C_properties = {
        'model_name': 'WX1282C',  # the model name
        'fw_ver': 0.0,  # the firmware version
        'serial_num': '0' * 9,  # serial number
        'num_parts': 2,  # number of instrument parts
        'chan_per_part': 1,  # number of channels per part
        'seg_quantum': 16,  # segment-length quantum
        'min_seg_len': 192,  # minimal segment length
        'max_arb_mem': 32E6,  # maximal arbitrary-memory (points per channel)
        'min_dac_val': 0,  # minimal DAC value
        'max_dac_val': 2 ** 14 - 1,  # maximal DAC value
        'max_num_segs': 32E+3,  # maximal number of segments
        'max_seq_len': 48 * 1024,  # maximal sequencer-table length (# rows)
        'min_seq_len': 3,  # minimal sequencer-table length (# rows)
        'max_num_seq': 1000,  # maximal number of sequencer-table
        'max_aseq_len': 1000,  # maximal advanced-sequencer table length
        'min_aseq_len': 3,  # minimal advanced-sequencer table length
        'min_sclk': 10e6,  # minimal sampling-rate (samples/seconds)
        'max_sclk': 1.25e9,  # maximal sampling-rate (samples/seconds)
        'digital_support': False,  # is digital-wave supported?
    }

    # dictionary of supported-models' properties
    model_properties_dict = {
        'WX2184': _wx2184_properties,
        'WX2184C': _wx2184_properties,
        'WX1284': _wx2184_properties,
        'WX1284C': _wx2184_properties,
        'WX2182C': _wx2182C_properties,
        'WX1282C': _wx1282C_properties,
    }
    class TEWXAwg:
        _make_combined_wave_calls = []

        def __init__(self, *args, paranoia_level=1, model='WX2184C', **kwargs):
            self.logged_commands = []
            self.logged_queries = []
            self._visa_inst = dummy_pyvisa.resources.MessageBasedResource()
            self.paranoia_level = paranoia_level
            self.dev_properties = dummy_teawg.model_properties_dict[model]

            self._download_segment_lengths_calls = []
            self._send_binary_data_calls = []
            self._download_adv_seq_table_calls = []
            self._download_sequencer_table_calls = []

        @property
        def is_simulator(self):
            return False
        @property
        def visa_inst(self):
            return self._visa_inst
        def send_cmd(self, *args, **kwargs):
            self.logged_commands.append((args, kwargs))
        def send_query(self, *args, **kwargs):
            return self._visa_inst.ask(*args, **kwargs)
        def download_segment_lengths(self, seg_len_list, pref='dummy_pref', paranoia_level='dummy_paranoia'):
            self._download_segment_lengths_calls.append((seg_len_list, pref, paranoia_level))
        def send_binary_data(self, pref, bin_dat, paranoia_level='dummy_paranoia'):
            self._send_binary_data_calls.append((pref, bin_dat, paranoia_level))
        def download_adv_seq_table(self, advanced_sequencer_table, pref=':ASEQ:DATA', paranoia_level=None):
            self._download_adv_seq_table_calls.append((advanced_sequencer_table, pref, paranoia_level))
        def download_sequencer_table(self, *args, **kwargs):
            self._download_sequencer_table_calls.append((args, kwargs))

        @staticmethod
        def make_combined_wave(wav1, wav2, dest_array, dest_array_offset=0, add_idle_pts=False, quantum=16):
            dummy_teawg.TEWXAwg._make_combined_wave_calls.append((wav1, wav2, dest_array, dest_array_offset, add_idle_pts, quantum))

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

