import unittest


class MyTest(unittest.TestCase):
    @unittest.skip
    def test_the_thing(self):
        exec_test()

with_alazar = True


def get_pulse():
    from qupulse.pulses import SequencePulseTemplate as SPT, \
        RepetitionPulseTemplate as RPT, FunctionPulseTemplate as FPT, MultiChannelPulseTemplate as MPT

    sine = FPT('U*sin(2*pi*t/tau)', 'tau', channel='out')
    marker_on = FPT('1', 'tau', channel='trigger')

    multi = MPT([sine, marker_on], {'tau', 'U'})
    multi.atomicity = True

    assert sine.defined_channels == {'out'}
    assert multi.defined_channels == {'out', 'trigger'}

    sine.add_measurement_declaration('meas', 0, 'tau')

    base = SPT([(multi, dict(tau='tau', U='U'), dict(meas='A')),
                (multi, dict(tau='tau', U='U'), dict(meas='A')),
                (multi, dict(tau='tau', U='U'), dict(meas='A'))], {'tau', 'U'})

    repeated = RPT(base, 'n')

    root = SPT([repeated, repeated, repeated], {'tau', 'n', 'U'})

    assert root.defined_channels == {'out', 'trigger'}

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
    from atsaverage.operations import RepAverage

    return [RepAverage(identifier='REP_A', maskID='A')]


def get_window(card):
    from atsaverage.gui import ThreadedStatusWindow
    window = ThreadedStatusWindow(card)
    window.start()
    return window


def exec_test():
    import time
    import numpy as np

    t = []
    names = []

    def tic(name):
        t.append(time.time())
        names.append(name)

    from qupulse.hardware.awgs_new_driver.tabor import TaborChannelPair, TaborAWGRepresentation
    tawg = TaborAWGRepresentation(r'USB0::0x168C::0x2184::0000216488::INSTR', reset=True)

    tchannelpair = TaborChannelPair(tawg, (1, 2), 'TABOR_AB')
    tawg.paranoia_level = 2

    # warnings.simplefilter('error', Warning)

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
        alazar.config = get_alazar_config()

        alazar.register_operations('test', get_operations())

        window = get_window(atsaverage.core.getLocalCard(1, 1))

        hardware_setup.register_dac(alazar)

    repeated = get_pulse()

    from qupulse.pulses.sequencing import Sequencer

    tic('init')
    sequencer = Sequencer()
    sequencer.push(repeated,
                   parameters=dict(n=10000, tau=1920, U=0.5),
                   channel_mapping={'out': 'TABOR_A', 'trigger': 'TABOR_A_MARKER'},
                   window_mapping=dict(A='A'))
    instruction_block = sequencer.build()

    tic('sequence')

    hardware_setup.register_program('test', instruction_block)

    tic('register')

    if with_alazar:
        from atsaverage.masks import PeriodicMask
        m = PeriodicMask()
        m.identifier = 'D'
        m.begin = 0
        m.end = 1
        m.period = 1
        m.channel = 0
        alazar._registered_programs['test'].masks.append(m)

    tic('per_mask')

    hardware_setup.arm_program('test')

    tic('arm')

    for d, name in zip(np.diff(np.asarray(t)), names[1:]):
        print(name, d)

    d = 1
