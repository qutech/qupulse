import unittest
import os
import sys
import copy
import os
import os.path
import json
from tempfile import TemporaryDirectory

srcPath = os.path.dirname(os.path.abspath(__file__)).rsplit('tests',1)[0] + 'src'
sys.path.insert(0,srcPath)

from pulses.Serializer import StorageBackend, FilesystemBackend, Serializer
from pulses.TablePulseTemplate import TablePulseTemplate
from pulses.SequencePulseTemplate import SequencePulseTemplate

class FileSystemBackendTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = TemporaryDirectory()
        self.cwd = os.getcwd()
        os.chdir(self.tmpdir.name)
        dirname = 'fsbackendtest'
        os.mkdir(dirname) # replace by temporary directory
        self.backend = FilesystemBackend(dirname)
        self.testdata = 'dshiuasduzchjbfdnbewhsdcuzd'
        self.identifier = 'some name'

    def tearDown(self):
        os.chdir(self.cwd)
        self.tmpdir.cleanup()

    def test_fsbackend(self):
        # first put the data
        self.backend.put(self.testdata, self.identifier)
        print('Testing backend')

        # then retrieve it again
        data = self.backend.get(self.identifier)
        self.assertEqual(data, self.testdata)

class SerializerTests(unittest.TestCase):
    def test_builtin_types(self):
        serializer = Serializer()
        testdata = dict(nested_dict=dict(a=5, b='some string'),
                        some_list = list(range(12)))
        result = json.dumps(testdata)
        result2 = serializer.serialize(testdata)
        self.assertEqual(result, result2)

    def test_extra_types(self):
        serializer = Serializer()
        testdata = dict(set=set(range(12)),
                        frozenset=frozenset(range(12)))
        converted_testdata = dict(set=list(range(12)),
                                  frozenset=list(range(12)))
        result = json.dumps(converted_testdata)
        result2 = serializer.serialize(testdata)
        self.assertEqual(result, result2)

    def test_TablePulseTemplate(self):
        tpt = TablePulseTemplate()
        tpt.add_entry(1,1)
        tpt.add_entry('time', 'voltage')
        serializer = Serializer()
        result = serializer.serialize(tpt)

    def test_identifier(self):
        tpt = TablePulseTemplate(identifier='tpt')
        tpt2 = TablePulseTemplate()
        tpt2.add_entry(1,1)
        subtemplates = [(tpt, {}), (tpt2, {})]
        seq = SequencePulseTemplate(subtemplates, [])
        serializer = Serializer()
        result = serializer.serialize(seq)
        import ipdb
        ipdb.set_trace()

    def test_SequencePulseTemplate(self):
        seq = SequencePulseTemplate([], [], identifier='sequence')
        serializer = Serializer()
        result = serializer.serialize(seq)


if __name__ == "__main__":
    unittest.main(verbosity=2)
