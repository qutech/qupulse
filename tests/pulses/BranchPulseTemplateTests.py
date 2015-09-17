import unittest
import os
import sys
from src.pulses.BranchPulseTemplate import BranchPulseTemplate


srcPath = os.path.dirname(os.path.abspath(__file__)).rsplit('tests',1)[0] + 'src'
sys.path.insert(0,srcPath)

class GenericBranchPulseTemplateTest(unittest.TestCase):
    pass