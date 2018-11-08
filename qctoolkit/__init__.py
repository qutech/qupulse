"""This is a (hopefully temporary) alias package to not break existing code. If you know a better way please change"""
import sys
import re
import pkgutil
import importlib
import warnings
import logging

qupulse = importlib.import_module('qupulse')
sys.modules[__name__] = qupulse

aliased = {}

""" import all subpackages and submodules to assert that
from qupulse.pulse import TablePT as T1
from qctoolkit.pulse import TablePT as T2
assert T1 is T2 
"""
for _, name, ispkg in pkgutil.walk_packages(qupulse.__path__, 'qupulse.'):
    alias = re.sub('^qupulse.', '%s.' % __name__, name)

    try:
        imported = importlib.import_module(name)
    except ImportError:
        warnings.warn('Could not import %s. The alias %s was NOT created.' % (name, alias))
        continue
    sys.modules[alias] = imported
    aliased[alias] = name

logging.info('Created module aliases:', aliased)
