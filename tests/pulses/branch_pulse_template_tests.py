import unittest
import os
import sys
from qctoolkit.pulses.branch_pulse_template import BranchPulseTemplate


srcPath = os.path.dirname(os.path.abspath(__file__)).rsplit('tests',1)[0] + 'qctoolkit'
sys.path.insert(0,srcPath)

class GenericBranchPulseTemplateTest(unittest.TestCase):
    pass