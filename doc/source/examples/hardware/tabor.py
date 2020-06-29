from qupulse.hardware.setup import HardwareSetup, PlaybackChannel, MarkerChannel
from qupulse.hardware.awgs.tabor import TaborChannelPair, TaborAWGRepresentation
import pyvisa


def add_tabor_to_hardware_setup(hardware_setup: HardwareSetup, tabor_address: str = None, name: str = 'TABOR'):
    def _find_tabor_address():
        known_instruments = pyvisa.ResourceManager().list_resources()

        _tabor_address = None
        for address in known_instruments:
            if r'0x168C::0x2184' in address:
                _tabor_address = address
                break
        if _tabor_address is None:
            raise RuntimeError('Could not locate TaborAWG')

        return _tabor_address

    if tabor_address is None:
        tabor_address = _find_tabor_address()

    tawg = TaborAWGRepresentation(tabor_address, reset=True)

    channel_pairs = []
    for pair_name in ('AB', 'CD'):
        channel_pair = getattr(tawg, 'channel_pair_%s' % pair_name)
        channel_pairs.append(channel_pair)

        for ch_i, ch_name in enumerate(pair_name):
            playback_name = '{name}_{ch_name}'.format(name=name, ch_name=ch_name)
            hardware_setup.set_channel(playback_name, PlaybackChannel(channel_pair, ch_i))
            hardware_setup.set_channel(playback_name + '_MARKER', MarkerChannel(channel_pair, ch_i))

    return tawg, channel_pairs
