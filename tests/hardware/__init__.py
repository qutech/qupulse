from tests.hardware.dummy_modules import import_package

use_dummy_tabor = True
if use_dummy_tabor:
    import_package('pytabor')
    import_package('pyvisa')
    import_package('teawg')

use_dummy_atsaverage = True
if use_dummy_atsaverage:
    import_package('atsaverage')
