from qupulse.hardware.setup import HardwareSetup, PlaybackChannel, MarkerChannel
from qupulse.pulses import PointPT, RepetitionPT, TablePT


#%%
""" Connect and setup to your AWG. Change awg_address to the address of your awg and awg_name to the name of 
your AWGs manufacturer (Zürich Instruments: ZI, TaborElectronics: Tabor).
"""

awg_name = 'TABOR'
awg_address = '127.0.0.1'

hardware_setup = HardwareSetup()

if awg_name == 'ZI':
    from qupulse.hardware.awgs.zihdawg import HDAWGRepresentation
    awg = HDAWGRepresentation(awg_address, 'USB')

    channel_pairs = []
    for pair_name in ('AB', 'CD', 'EF', 'GH'):
        channel_pair = getattr(awg, 'channel_pair_%s' % pair_name)

        for ch_i, ch_name in enumerate(pair_name):
            playback_name = '{name}_{ch_name}'.format(name=awg_name, ch_name=ch_name)
            hardware_setup.set_channel(playback_name,
                                       PlaybackChannel(channel_pair, ch_i))
            hardware_setup.set_channel(playback_name + '_MARKER_FRONT', MarkerChannel(channel_pair, 2 * ch_i))
            hardware_setup.set_channel(playback_name + '_MARKER_BACK', MarkerChannel(channel_pair, 2 * ch_i + 1))
    awg_channel = awg.channel_pair_AB

elif awg_name == 'TABOR':
    from qupulse.hardware.awgs.tabor import TaborAWGRepresentation
    awg = TaborAWGRepresentation(awg_address, reset=True)

    channel_pairs = []
    for pair_name in ('AB', 'CD'):
        channel_pair = getattr(awg, 'channel_pair_%s' % pair_name)
        channel_pairs.append(channel_pair)

        for ch_i, ch_name in enumerate(pair_name):
            playback_name = '{name}_{ch_name}'.format(name=awg_name, ch_name=ch_name)
            hardware_setup.set_channel(playback_name, PlaybackChannel(channel_pair, ch_i))
            hardware_setup.set_channel(playback_name + '_MARKER', MarkerChannel(channel_pair, ch_i))
    awg_channel = channel_pairs[0]

else:
    ValueError('Unknown AWG')

#%%
""" Create three simple pulses and put them together to a PulseTemplate called dnp """

plus = [(0, 0), ('ta', 'va', 'hold'), ('tb', 'vb', 'linear'), ('tend', 0, 'jump')]
minus = [(0, 0), ('ta', '-va', 'hold'), ('tb', '-vb', 'linear'), ('tend', 0, 'jump')]

zero_pulse = PointPT([(0, 0), ('tend', 0)], ('X', 'Y'))
plus_pulse = TablePT(entries={'X': plus, 'Y': plus})
minus_pulse = TablePT(entries={'X': minus, 'Y': minus})

dnp = RepetitionPT(minus_pulse, 'n_minus') @ RepetitionPT(zero_pulse, 'n_zero') @ RepetitionPT(plus_pulse, 'n_plus')

#%%
""" Create a program dnp with the number of pulse repetitions as volatile parameters """

sample_rate = awg_channel.sample_rate / 10**9
n_quant = 192
t_quant = n_quant / sample_rate

dnp_prog = dnp.create_program(parameters=dict(tend=float(t_quant), ta=float(t_quant/3), tb=float(2*t_quant/3),
                                              va=0.12, vb=0.25, n_minus=3, n_zero=3, n_plus=3),
                              channel_mapping={'X': '{}_A'.format(awg_name), 'Y': '{}_B'.format(awg_name)},
                              volatile={'n_minus', 'n_zero', 'n_plus'})
dnp_prog.cleanup()

#%%
""" Upload this program to the AWG """

hardware_setup.register_program('dnp', dnp_prog)
hardware_setup.arm_program('dnp')

#%%
""" Run initial program """

awg_channel.run_current_program()

#%%
""" Change volatile parameters to new values and run the modified program """

hardware_setup.update_parameters('dnp', dict(n_zero=1, n_plus=5))
awg_channel.run_current_program()
