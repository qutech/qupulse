import sys
import re
import pkgutil
import importlib

qupulse = importlib.import_module('qupulse')
sys.modules[__name__] = qupulse

aliased = {}

for _, name, ispkg in pkgutil.walk_packages(qupulse.__path__, 'qupulse.'):
    alias = re.sub('^qupulse.', '%s.' % __name__, name)

    try:
        imported = importlib.import_module(name)
    except ImportError:
        print('Could not import %s. The alias %s was NOT created.' % (name, alias))
        continue
    sys.modules[alias] = imported
    aliased[alias] = name

print('Created module aliases:', aliased)
