import unittest


with_alazar = True

def get_pulse():
    from qupulse.pulses import TablePulseTemplate as TPT, SequencePulseTemplate as SPT, RepetitionPulseTemplate as RPT

    ramp = TPT(identifier='ramp', channels={'out', 'trigger'})
    ramp.add_entry(0, 'start', channel='out')
    ramp.add_entry('duration', 'stop', 'linear', channel='out')

    ramp.add_entry(0, 1, channel='trigger')
    ramp.add_entry('duration', 1, 'hold', channel='trigger')

    ramp.add_measurement_declaration('meas', 0, 'duration')

    base = SPT([(ramp, dict(start='min', stop='max', duration='tau/3'), dict(meas='A')),
                (ramp, dict(start='max', stop='max', duration='tau/3'), dict(meas='B')),
                (ramp, dict(start='max', stop='min', duration='tau/3'), dict(meas='C'))], {'min', 'max', 'tau'})

    repeated = RPT(base, 'n')

    root = SPT([repeated, repeated, repeated], {'min', 'max', 'tau', 'n'})

    return root


def get_alazar_config():
    from atsaverage import alazar
    from atsaverage.config import ScanlineConfiguration, CaptureClockConfiguration, EngineTriggerConfiguration,\
        TRIGInputConfiguration, InputConfiguration

    trig_level = int((5 + 0.4) / 10. * 255)
    assert 0 <= trig_level < 256

    config = ScanlineConfiguration()
    config.triggerInputConfiguration = TRIGInputConfiguration(triggerRange=alazar.TriggerRangeID.etr_5V)
    config.triggerConfiguration = EngineTriggerConfiguration(triggerOperation=alazar.TriggerOperation.J,
                                                             triggerEngine1=alazar.TriggerEngine.J,
                                                             triggerSource1=alazar.TriggerSource.external,
                                                             triggerSlope1=alazar.TriggerSlope.positive,
                                                             triggerLevel1=trig_level,
                                                             triggerEngine2=alazar.TriggerEngine.K,
                                                             triggerSource2=alazar.TriggerSource.disable,
                                                             triggerSlope2=alazar.TriggerSlope.positive,
                                                             triggerLevel2=trig_level)
    config.captureClockConfiguration = CaptureClockConfiguration(source=alazar.CaptureClockType.internal_clock,
                                                                 samplerate=alazar.SampleRateID.rate_100MSPS)
    config.inputConfiguration = 4*[InputConfiguration(input_range=alazar.InputRangeID.range_1_V)]
    config.totalRecordSize = 0

    assert config.totalRecordSize == 0

    return config

def get_operations():
    from atsaverage.operations import Downsample

    return [Downsample(identifier='DS_A', maskID='A'),
            Downsample(identifier='DS_B', maskID='B'),
            Downsample(identifier='DS_C', maskID='C'),
            Downsample(identifier='DS_D', maskID='D')]

def get_window(card):
    from atsaverage.gui import ThreadedStatusWindow
    window = ThreadedStatusWindow(card)
    window.start()
    return window


class TaborTests(unittest.TestCase):
    @unittest.skip
    def test_all(self):
        from qupulse.hardware.awgs_new_driver.tabor import TaborChannelTuple, TaborDevice
        #import warnings
        tawg = TaborDevice(r'USB0::0x168C::0x2184::0000216488::INSTR')
        tchannelpair = TaborChannelTuple(tawg, (1, 2), 'TABOR_AB')
        tawg.paranoia_level = 2

        #warnings.simplefilter('error', Warning)

        from qupulse.hardware.setup import HardwareSetup, PlaybackChannel, MarkerChannel
        hardware_setup = HardwareSetup()

        hardware_setup.set_channel('TABOR_A', PlaybackChannel(tchannelpair, 0))
        hardware_setup.set_channel('TABOR_B', PlaybackChannel(tchannelpair, 1))
        hardware_setup.set_channel('TABOR_A_MARKER', MarkerChannel(tchannelpair, 0))
        hardware_setup.set_channel('TABOR_B_MARKER', MarkerChannel(tchannelpair, 1))

        if with_alazar:
            from qupulse.hardware.dacs.alazar import AlazarCard
            import atsaverage.server

            if not atsaverage.server.Server.default_instance.running:
                atsaverage.server.Server.default_instance.start(key=b'guest')

            import atsaverage.core

            alazar = AlazarCard(atsaverage.core.getLocalCard(1, 1))
            alazar.register_mask_for_channel('A', 0)
            alazar.register_mask_for_channel('B', 0)
            alazar.register_mask_for_channel('C', 0)
            alazar.config = get_alazar_config()

            alazar.register_operations('test', get_operations())
            window = get_window(atsaverage.core.getLocalCard(1, 1))
            hardware_setup.register_dac(alazar)

        repeated = get_pulse()

        from qupulse.pulses.sequencing import Sequencer

        sequencer = Sequencer()
        sequencer.push(repeated,
                       parameters=dict(n=1000, min=-0.5, max=0.5, tau=192*3),
                       channel_mapping={'out': 'TABOR_A', 'trigger': 'TABOR_A_MARKER'},
                       window_mapping=dict(A='A', B='B', C='C'))
        instruction_block = sequencer.build()

        hardware_setup.register_program('test', instruction_block)

        if with_alazar:
            from atsaverage.masks import PeriodicMask
            m = PeriodicMask()
            m.identifier = 'D'
            m.begin = 0
            m.end = 1
            m.period = 1
            m.channel = 0
            alazar._registered_programs['test'].masks.append(m)

        hardware_setup.arm_program('test')

        d = 1

