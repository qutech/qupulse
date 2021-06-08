"""Import dummy packages for non-available drivers"""
from tests.hardware.dummy_modules import import_package

try:
    import atsaverage
except ImportError:
    atsaverage = import_package('atsaverage')
