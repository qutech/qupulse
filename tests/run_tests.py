import sys
import os
import unittest

#basePath = os.path.dirname(os.path.abspath(__file__))
#sys.path.insert(0, basePath)

suite = unittest.TestLoader().discover('.', pattern='*Test*.py')
unittest.TextTestRunner(verbosity=2).run(suite)
