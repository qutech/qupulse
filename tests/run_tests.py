import sys
import os
import unittest

#basePath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#srcPath = basePath + '\\src'
#sys.path.insert(0, srcPath)

suite = unittest.TestLoader().discover('.', pattern='*Test*.py')

unittest.TextTestRunner(verbosity=2).run(suite)
