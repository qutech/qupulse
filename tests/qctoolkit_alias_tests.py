import unittest


class QctoolkitAliasTest(unittest.TestCase):
    def test_alias(self):
        import qctoolkit.pulses
        import qupulse.pulses

        self.assertIs(qctoolkit.pulses, qupulse.pulses)
        self.assertIs(qctoolkit.pulses.TablePT, qupulse.pulses.TablePT)

    def test_class_identity(self):
        from qupulse._program._loop import Loop as Loop_qu
        from qctoolkit._program._loop import Loop as Loop_qc

        self.assertIs(Loop_qc, Loop_qu)
