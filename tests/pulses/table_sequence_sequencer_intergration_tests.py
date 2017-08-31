import unittest

from qctoolkit.pulses.multi_channel_pulse_template import MappingPulseTemplate
from qctoolkit.pulses.table_pulse_template import TablePulseTemplate
from qctoolkit.pulses.sequence_pulse_template import SequencePulseTemplate
from qctoolkit.pulses.parameters import ParameterNotProvidedException
from qctoolkit.pulses.sequencing import Sequencer
from qctoolkit.pulses.instructions import EXECInstruction, AbstractInstructionBlock

from tests.pulses.sequencing_dummies import DummyParameter, DummyNoValueParameter


class TableSequenceSequencerIntegrationTests(unittest.TestCase):

    def test_table_sequence_sequencer_integration(self) -> None:
        t1 = TablePulseTemplate(entries={'default': [(2, 'foo'),
                                                     (5, 0)]},
                                measurements=[('foo', 2, 2)])

        t2 = TablePulseTemplate(entries={'default': [(4, 0),
                                                     (4.5, 'bar', 'linear'),
                                                     (5, 0)]},
                                measurements=[('foo', 4, 1)])

        seqt = SequencePulseTemplate(MappingPulseTemplate(t1, measurement_mapping={'foo': 'bar'}),
                                     MappingPulseTemplate(t2, parameter_mapping={'bar': '2 * hugo'}))

        with self.assertRaises(ParameterNotProvidedException):
            t1.requires_stop(dict(), dict())
        with self.assertRaises(ParameterNotProvidedException):
            t2.requires_stop(dict(), dict())
        self.assertFalse(seqt.requires_stop({'foo': DummyParameter(), 'hugo': DummyParameter()}, {}))

        foo = DummyNoValueParameter()
        bar = DummyNoValueParameter()
        sequencer = Sequencer()
        sequencer.push(seqt, {'foo': foo, 'hugo': bar},
                       window_mapping=dict(bar='my', foo='thy'),
                       channel_mapping={'default': 'A'})
        instructions = sequencer.build()
        self.assertFalse(sequencer.has_finished())
        self.assertEqual(1, len(instructions))

        foo = DummyParameter(value=1.1)
        bar = DummyNoValueParameter()
        sequencer = Sequencer()
        sequencer.push(seqt, {'foo': foo, 'hugo': bar},
                       window_mapping=dict(bar='my', foo='thy'),
                       channel_mapping={'default': 'A'})
        instructions = sequencer.build()
        self.assertFalse(sequencer.has_finished())
        self.assertIsInstance(instructions, AbstractInstructionBlock)
        self.assertEqual(2, len(instructions))

        foo = DummyParameter(value=1.1)
        bar = DummyNoValueParameter()
        sequencer = Sequencer()
        sequencer.push(seqt, {'foo': bar, 'hugo': foo},
                       window_mapping=dict(bar='my', foo='thy'),
                       channel_mapping={'default': 'A'})
        instructions = sequencer.build()
        self.assertFalse(sequencer.has_finished())
        self.assertEqual(1, len(instructions))

        foo = DummyParameter(value=1.1)
        bar = DummyParameter(value=-0.2)
        sequencer = Sequencer()
        sequencer.push(seqt, {'foo': foo, 'hugo': bar},
                       window_mapping=dict(bar='my', foo='thy'),
                       channel_mapping={'default': 'A'})
        instructions = sequencer.build()
        self.assertTrue(sequencer.has_finished())
        self.assertEqual(3, len(instructions))

        for instruction in instructions:
            if isinstance(instruction, EXECInstruction):
                self.assertIn(instruction.waveform.get_measurement_windows()[0], [('my', 2, 2), ('thy', 4, 1)])
