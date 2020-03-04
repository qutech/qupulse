from qupulse.hardware.awgs.zihdawg import HDAWGRepresentation
from qupulse.hardware.setup import HardwareSetup, PlaybackChannel, MarkerChannel


def add_to_hardware_setup(hardware_setup: HardwareSetup, serial, name='ZI'):
    hdawg = HDAWGRepresentation(serial, 'USB')

    channel_pairs = []
    for pair_name in ('AB', 'CD', 'EF', 'GH'):
        channel_pair = getattr(hdawg, 'channel_pair_%s' % pair_name)

        for ch_i, ch_name in enumerate(pair_name):
            playback_name = '{name}_{ch_name}'.format(name=name, ch_name=ch_name)
            hardware_setup.set_channel(playback_name,
                                       PlaybackChannel(channel_pair, ch_i))
            hardware_setup.set_channel(playback_name + '_MARKER_FRONT', MarkerChannel(channel_pair, 2 * ch_i))
            hardware_setup.set_channel(playback_name + '_MARKER_BACK', MarkerChannel(channel_pair, 2 * ch_i + 1))

    return hdawg, channel_pairs
