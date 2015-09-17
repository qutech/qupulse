import unittest
import sys
import os
import os.path
import json
from tempfile import TemporaryDirectory

srcPath = os.path.dirname(os.path.abspath(__file__)).rsplit('tests',1)[0] + 'src'
sys.path.insert(0,srcPath)

from pulses.Serializer import FilesystemBackend, Serializer
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

        # then retrieve it again
        data = self.backend.get(self.identifier)
        self.assertEqual(data, self.testdata)


class SerializerTests(unittest.TestCase):
    pass

    #def test_serialization(self) -> None:
        #table_foo = TablePulseTemplate(identifier='foo')
        #table = TablePulseTemplate(measurement=True)
        #sequence = SequencePulseTemplate([(table_foo, {}), (table, {})], [], identifier=None)
        #serializer = Serializer(FilesystemBackend())
        #serializer.serialize(sequence)
        #self.fail() # TODO: instead of printing, compare against expected values
        #print(json.dumps(serialized, indent=4))

    # def test_TablePulseTemplate(self):
    #     tpt = TablePulseTemplate()
    #     tpt.add_entry(1,1)
    #     tpt.add_entry('time', 'voltage')
    #     serializer = Serializer()
    #     result = serializer.serialize(tpt)
    #     self.fail() # TODO: this test was not complete when merged
    #
    # def test_SequencePulseTemplate(self):
    #     seq = SequencePulseTemplate([], [], identifier='sequence')
    #     serializer = Serializer()
    #     result = serializer.serialize(seq)
    #     self.fail() # TODO: this test was not complete when merged


if __name__ == "__main__":
    unittest.main(verbosity=2)
