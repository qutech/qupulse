import pickle
from qupulse.serialization import PulseStorage, FilesystemBackend
from qupulse.hardware.awgs.tabor import TaborAWGRepresentation
from qupulse._program._loop import _is_compatible, Fraction

with open(r'Y:\GaAs\Humpohl\temp.bin', 'rb') as f:
    kwargs = pickle.loads(f.read())

ps = PulseStorage(FilesystemBackend(r'Y:\GaAs\Humpohl'))


pulse = ps['charge_4chan']


program = pulse.create_program(**kwargs)

_is_compatible(program, 192, 16, Fraction(2))

"""
def prepare(p):
    return p, p.copy_tree_structure()


p_AB, p_CD = prepare(program)

tawg = TaborAWGRepresentation('TCPIP::127.0.0.1::5025::SOCKET')

tawg.send_cmd(':FREQ:RAST 2e9')

tawg.channel_pair_AB.upload('test', p_AB,
                            (None, None), markers=('TABOR_A_MARKER', 'TABOR_B_MARKER'),
                            voltage_transformation=(lambda x: x, lambda x: x))
tawg.channel_pair_CD.upload('test', p_CD,
                            ('TABOR_C', 'TABOR_D'), markers=('TABOR_C_MARKER', None),
                            voltage_transformation=(lambda x: x, lambda x: x))
"""