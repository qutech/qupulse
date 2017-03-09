"""Import dummy modules if actual modules not installed. Sets dummy modules in sys so subsequent imports
use the dummies"""

import sys

class dummy_pytabor:
    pass

class dummy_pyvisa:
    pass

class dummy_teawg:
    model_properties_dict = dict()
    class TEWXAwg:
        def __init__(self, *args, **kwargs):
            pass
        send_cmd = __init__
        send_query = send_cmd
        select_channel = send_cmd
        send_binary_data = send_cmd
        download_sequencer_table = send_cmd

failed_imports = set()
try:
    import pytabor
except ImportError:
    sys.modules['pytabor'] = dummy_pytabor
    failed_imports.add(dummy_pytabor)

try:
    import pyvisa
except ImportError:
    sys.modules['pyvisa'] = dummy_pyvisa
    failed_imports.add(dummy_pyvisa)

try:
    import teawg
except ImportError:
    sys.modules['teawg'] = dummy_teawg
    failed_imports.add(dummy_teawg)
