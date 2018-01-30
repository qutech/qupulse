"""Import dummy packages for non-available drivers"""
from tests.hardware.dummy_modules import import_package

try:
    import atsaverage
except ImportError:
    atsaverage = import_package('atsaverage')

try:
    import pyvisa
except ImportError:
    pyvisa = import_package('pyvisa')

try:
    import pytabor
except ImportError:
    pytabor = import_package('pytabor')

try:
    import teawg
except ImportError:
    teawg = import_package('teawg')