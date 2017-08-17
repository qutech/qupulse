
from . import dummy_modules


use_dummy_tabor = True
if use_dummy_tabor:
    dummy_modules.import_package('pytabor', dummy_modules.dummy_pytabor)
    dummy_modules.import_package('pyvisa', dummy_modules.dummy_pyvisa)
    dummy_modules.import_package('teawg', dummy_modules.dummy_teawg)

use_dummy_atsaverage = True
if use_dummy_atsaverage:
    dummy_modules.import_package('atsaverage')
