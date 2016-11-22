import unittest

from qctoolkit.pulses.pulse_template import SubTemplate
from qctoolkit.pulses.table_pulse_template import TablePulseTemplate
from qctoolkit.pulses.sequence_pulse_template import SequencePulseTemplate
from qctoolkit.pulses.parameters import ParameterNotProvidedException
from qctoolkit.pulses.sequencing import Sequencer
from qctoolkit.pulses.instructions import EXECInstruction

from tests.pulses.sequencing_dummies import DummyParameter, DummyNoValueParameter


class TableSequenceSequencerIntegrationTests(unittest.TestCase):

    def test_table_sequence_sequencer_integration(self) -> None:
        t1 = TablePulseTemplate()
        t1.add_entry(2, 'foo')
        t1.add_entry(5, 0)
        t1.add_measurement_declaration('foo', 2, 5)

        t2 = TablePulseTemplate()
        t2.add_entry(4, 0)
        t2.add_entry(4.5, 'bar', 'linear')
        t2.add_entry(5, 0)
        t2.add_measurement_declaration('foo', 4, 5)

        seqt = SequencePulseTemplate([SubTemplate(t1, {'foo': 'foo'}, measurement_mapping={'foo': 'bar'}),
                                      SubTemplate(t2, {'bar': '2 * hugo'})], {'foo', 'hugo'})

        with self.assertRaises(ParameterNotProvidedException):
            t1.requires_stop(dict(), dict())
        with self.assertRaises(ParameterNotProvidedException):
            t2.requires_stop(dict(), dict())
        self.assertFalse(seqt.requires_stop({}, {}))

        foo = DummyNoValueParameter()
        bar = DummyNoValueParameter()
        sequencer = Sequencer()
        sequencer.push(seqt, {'foo': foo, 'hugo': bar}, window_mapping=dict(bar='my', foo='thy'))
        instructions = sequencer.build()
        self.assertFalse(sequencer.has_finished())
        self.assertEqual(1, len(instructions))

        foo = DummyParameter(value=1.1)
        bar = DummyNoValueParameter()
        sequencer = Sequencer()
        sequencer.push(seqt, {'foo': foo, 'hugo': bar}, window_mapping=dict(bar='my', foo='thy'))
        instructions = sequencer.build()
        self.assertFalse(sequencer.has_finished())
        self.assertEqual(2, len(instructions))

        foo = DummyParameter(value=1.1)
        bar = DummyNoValueParameter()
        sequencer = Sequencer()
        sequencer.push(seqt, {'foo': bar, 'hugo': foo}, window_mapping=dict(bar='my', foo='thy'))
        instructions = sequencer.build()
        self.assertFalse(sequencer.has_finished())
        self.assertEqual(1, len(instructions))

        foo = DummyParameter(value=1.1)
        bar = DummyParameter(value=-0.2)
        sequencer = Sequencer()
        sequencer.push(seqt, {'foo': foo, 'hugo': bar}, window_mapping=dict(bar='my', foo='thy'))
        instructions = sequencer.build()
        self.assertTrue(sequencer.has_finished())
        self.assertEqual(3, len(instructions))

        for instruction in instructions:
            if isinstance(instruction,EXECInstruction):
                self.assertIn(instruction.measurement_windows[0], [('my', 2, 5),('thy', 4, 5)])
