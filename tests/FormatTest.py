import unittest
import os
import sys

srcPath = os.path.dirname(os.path.abspath(__file__)).rsplit('tests',1)[0] + 'src'
sys.path.insert(0,srcPath)

@unittest.skipIf(os.getenv('TRAVIS', False), "skipping format tests on travis")
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

@unittest.skipIf(os.getenv('TRAVIS', False), "skipping format tests on travis")
class DocumentationTest(unittest.TestCase):
    def _test_attribute(self,package,name):
        try:
            bool = hasattr(__import__(package, fromlist=[name]), name)
        except ImportError as e:
            return False
        else:
            return bool
        
    def test_documentation_in_all_py_files(self,raiseonerror=True):
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
                            for method in methods[name]:
                                if self._test_attribute(package,method):
                                    imported = getattr(__import__(package, fromlist=[method]), method)
                                    if raiseonerror:
                                        self.assertIsNotNone(imported.__doc__,"The docstring should not be empty. Module: {}, Method: {}".format(package,method))
                                    else:
                                        if imported.__doc__ is None:
                                            print("The docstring should not be empty. Module: {}, Method: {}".format(package,method))
                        else:
                            if self._test_attribute(package,name):
                                imported = getattr(__import__(package, fromlist=[name]), name)
                                if raiseonerror:
                                    self.assertIsNotNone(imported.__doc__,"The docstring should not be empty. Module: {}, Class: {}".format(package,name))
                                else:
                                    if imported.__doc__ is None:
                                        print("The docstring should not be empty. Module: {}, Class: {}".format(package,name))
                                for method in methods[name]:
                                    if hasattr(imported, method):
                                        if raiseonerror:
                                            self.assertIsNotNone(getattr(imported, method).__doc__, "The docstring should not be empty. Module: {}, Class: {}, Method: {}".format(package,name,method))
                                        else:
                                            if getattr(imported, method).__doc__ is None:
                                                print("The docstring should not be empty. Module: {}, Class: {}, Method: {}".format(package,name,method))
                                        
@unittest.skipIf(os.getenv('TRAVIS', False), "skipping format tests on travis")
class ImportTest(unittest.TestCase):
    def test_imports(self):
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
                            for method in methods[name]:
                                try:
                                    hasattr(__import__(package, fromlist=[method]), method)
                                except Exception as e:
                                    raise Exception("{} in file {}.".format(str(e),file_path)) from e
                        else:
                            try:
                                isobject = hasattr(__import__(package, fromlist=[name]), name)
                            except Exception as e:
                                raise Exception("{} in file {}.".format(str(e),file_path)) from e
                            else:
                                if isobject:
                                    for method in methods[name]:
                                        try:
                                            hasattr(method, name)
                                        except Exception as e:
                                            raise Exception("{} in file {}.".format(str(e),file_path)) from e
        
@unittest.skipIf(os.getenv('TRAVIS', False), "skipping format tests on travis")
class AnnotationTest(unittest.TestCase):
    def _test_attribute(self,package,name):
        try:
            bool = hasattr(__import__(package, fromlist=[name]), name)
        except ImportError as e:
            return False
        else:
            return bool
        
    def test_annotations(self):
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
                            for method in methods[name]:
                                if self._test_attribute(package,method):
                                    imported = getattr(__import__(package, fromlist=[method]), method)
                                    self.assertIsNotNone(imported.__annotations__,"No Annotation found. Module: {}, Method: {}".format(package,method))
                        else:
                            if self._test_attribute(package,name):
                                imported = getattr(__import__(package, fromlist=[name]), name)
                                for method in methods[name]:
                                    if hasattr(imported, method):
                                        loaded_method = getattr(imported, method)
                                        if hasattr(loaded_method,"__call__"):
                                            self.assertIn("return",loaded_method.__annotations__,"No Return annotation found for Module: {}, Class: {}, Method: {}".format(package,name,method))
                                            self.assertNotEqual(len(loaded_method.__annotations__.keys()),0,"No Annotation found. Module: {}, Class: {}, Method: {}".format(package,name,method))
