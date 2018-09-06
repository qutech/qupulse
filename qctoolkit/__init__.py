import sys
import importlib

sys.modules[__name__] = importlib.import_module('qupulse')
