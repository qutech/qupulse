import unittest
import os
import sys

srcPath = os.path.dirname(os.path.abspath(__file__)).rsplit('tests', 1)[0] + 'qctoolkit'
sys.path.insert(0, srcPath)

class TabTest(unittest.TestCase):
    def test_tabs_in_all_py_files(self):    
        for root, dirs, files in os.walk(srcPath):
            for filename in files:
                file_path = os.path.join(root, filename)
                if file_path.endswith(".py"):
                    with open(file_path) as infile:
                        for index, line in enumerate(infile):
                            self.assertNotIn(u"\t", line, "Tabs should be replaced by 4 spaces. File: {}, Line: {}".format(file_path, index))
