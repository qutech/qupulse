import os
import sys
"""Change the path as we were in the similar path in the src directory"""
srcPath = "src".join(os.path.dirname(os.path.abspath(__file__)).rsplit('tests',1))
sys.path.insert(0,srcPath)

import unittest

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

class DocumentationTest(unittest.TestCase):
    def _test_attribute(self,package,name):
        try:
            bool = hasattr(__import__(package, fromlist=[name]), name)
        except ImportError as e:
            return False
        else:
            return bool
        
    def test_documentation_in_all_py_files(self):
        raiseonerror = True
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
                                if hasattr(__import__(package, fromlist=[method]), method):
                                    pass
                        else:
                            if hasattr(__import__(package, fromlist=[name]), name):
                                for method in methods[name]:
                                    if hasattr(imported, method):
                                        pass
                                    
if __name__ == "__main__":
    unittest.main()