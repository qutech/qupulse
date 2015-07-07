import sys
import pyflakes.api
from pyflakes.reporter import Reporter

pyflakes.api.checkRecursive(['.'], Reporter(sys.stdout, sys.stderr))
