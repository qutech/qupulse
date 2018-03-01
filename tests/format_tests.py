import unittest
import os
import sys

srcPath = os.path.dirname(os.path.abspath(__file__)).rsplit('tests',1)[0] + 'qctoolkit'
sys.path.insert(0,srcPath)

class TabTest(unittest.TestCase):
    def test_tabs_in_all_py_files(self):    
        for root, dirs, files in os.walk(srcPath):
            for filename in files:
                #
                file_path = os.path.join(root,filename)
                if file_path.endswith("py"):
                    with open(file_path) as infile:
                        for index,line in enumerate(infile):
                            self.assertNotIn(u"\t",line,"Tabs should be replaced by 4 spaces. File: {}, Line: {}".format(file_path, index))

class AnnotationTest(unittest.TestCase):
    def _test_attribute(self,package,name):
        try:
            bool = hasattr(__import__(package, fromlist=[name]), name)
        except ImportError:
            return False
        else:
            return bool
        
    def test_annotations(self):
        whitelist = ["__init__", "__new__", "__str__", "__repr__", "__hash__"]
        for root, dirs, files in os.walk(srcPath):
            for filename in files:
                methods = {}
                file_path = os.path.join(root,filename)
                if file_path.endswith("py"):
                    with open(file_path) as infile:
                        inClass = "None"
                        if inClass not in methods:
                            methods[inClass] = []
                        for index,line in enumerate(infile):
                            if "class" in line.split():
                                inClass = (line.split()[1].split("(")[0])
                                if inClass not in methods:
                                    methods[inClass] = []
                            elif "def" in line.split():
                                methods[inClass].append(line.split()[1].split("(")[0])
                    package = (".".join(os.path.join(root,filename.split(".")[0])[len(srcPath)+1:].split(os.path.sep)))
                    for name in methods:
                        if name == "None":
                            for method in methods[name] :
                                if method not in whitelist:
                                    if self._test_attribute(package,method):
                                        imported = getattr(__import__(package, fromlist=[method]), method)
                                        self.assertIsNotNone(imported.__annotations__,"No Annotation found. Module: {}, Method: {}".format(package,method))
                        else:
                            if self._test_attribute(package,name):
                                imported = getattr(__import__(package, fromlist=[name]), name)
                                for method in methods[name]:
                                    if method not in whitelist:
                                        if hasattr(imported, method):
                                            loaded_method = getattr(imported, method)
                                            if hasattr(loaded_method, "__call__") and hasattr(loaded_method,
                                                                                              "__annotations__"):
                                                self.assertIn("return",loaded_method.__annotations__,"No Return annotation found for Module: {}, Class: {}, Method: {}".format(package,name,method))
                                                self.assertNotEqual(len(loaded_method.__annotations__.keys()),0,"No Annotation found. Module: {}, Class: {}, Method: {}".format(package,name,method))
