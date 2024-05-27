# SPDX-FileCopyrightText: 2014-2024 Quantum Technology Group and Chair of Software Engineering, RWTH Aachen University
#
# SPDX-License-Identifier: GPL-3.0-or-later

import sys
import argparse
import logging

from typing import Set

if sys.version_info.minor > 8:
    try:
        from qupulse_hdawg.zihdawg import *
    except ImportError:
        print("Install the qupulse_hdawg package to use HDAWG with this python version.")
        raise
else:
    try:
        from qupulse_hdawg_legacy.zihdawg import *
    except ImportError:
        print("Install the qupulse_hdawg_legacy package to use HDAWG with this python version.")
        raise


def example_upload(hdawg_kwargs: dict, channels: Set[int], markers: Set[Tuple[int, int]]):  # pragma: no cover
    from qupulse.pulses import TablePT, SequencePT, RepetitionPT
    if isinstance(hdawg_kwargs, dict):
        hdawg = HDAWGRepresentation(**hdawg_kwargs)
    else:
        hdawg = hdawg_kwargs

    assert not set(channels) - set(range(8)), "Channels must be in 0..=7"
    channels = sorted(channels)

    required_channels = {*channels, *(ch for ch, _ in markers)}
    channel_group = get_group_for_channels(hdawg, required_channels)
    channel_group_channels = range(channel_group.awg_group_index * channel_group.num_channels,
                                   (channel_group.awg_group_index + 1) * channel_group.num_channels)

    # choose length based on minimal sample rate
    sample_rate = channel_group.sample_rate / 10**9
    min_t = channel_group.MIN_WAVEFORM_LEN / sample_rate
    quant_t = channel_group.WAVEFORM_LEN_QUANTUM / sample_rate

    assert min_t > 4 * quant_t, "Example not updated"

    entry_list1 = [(0, 0), (quant_t * 2, .2, 'hold'),    (min_t,  .3, 'linear'),   (min_t + 3*quant_t, 0, 'jump')]
    entry_list2 = [(0, 0), (quant_t * 3, -.2, 'hold'),   (min_t, -.3, 'linear'),  (min_t + 4*quant_t, 0, 'jump')]
    entry_list3 = [(0, 0), (quant_t * 1, -.2, 'linear'), (min_t, -.3, 'linear'), (2*min_t, 0, 'jump')]
    entry_lists = [entry_list1, entry_list2, entry_list3]

    entry_dict1 = {ch: entry_lists[:2][i % 2] for i, ch in enumerate(channels)}
    entry_dict2 = {ch: entry_lists[1::-1][i % 2] for i, ch in enumerate(channels)}
    entry_dict3 = {ch: entry_lists[2:0:-1][i % 2] for i, ch in enumerate(channels)}

    tpt1 = TablePT(entry_dict1, measurements=[('m', 20, 30)])
    tpt2 = TablePT(entry_dict2)
    tpt3 = TablePT(entry_dict3, measurements=[('m', 10, 50)])
    rpt = RepetitionPT(tpt1, 4)
    spt = SequencePT(tpt2, rpt)
    rpt2 = RepetitionPT(spt, 2)
    spt2 = SequencePT(rpt2, tpt3)
    p = spt2.create_program()

    upload_ch = tuple(ch if ch in channels else None
                      for ch in channel_group_channels)
    upload_mk = (None,) * channel_group.num_markers
    upload_vt = (lambda x: x,) * channel_group.num_channels

    channel_group.upload('pulse_test1', p, upload_ch, upload_mk, upload_vt)

    if markers:
        markers = sorted(markers)
        assert len(markers) == len(set(markers))
        channel_group_markers = tuple((ch, mk)
                                      for ch in channel_group_channels
                                      for mk in (0, 1))

        full_on = [(0, 1), (min_t, 1)]
        two_3rd = [(0, 1), (min_t*2/3, 0), (min_t, 0)]
        one_3rd = [(0, 0), (min_t*2/3, 1), (min_t, 1)]

        marker_start = TablePT({'m0': full_on, 'm1': full_on})
        marker_body = TablePT({'m0': two_3rd, 'm1': one_3rd})

        marker_test_pulse = marker_start @ RepetitionPT(marker_body, 10000)

        marker_program = marker_test_pulse.create_program()

        upload_ch = (None, ) * channel_group.num_channels
        upload_mk = tuple(f"m{mk}" if (ch, mk) in markers else None
                          for (ch, mk) in channel_group_markers)

        channel_group.upload('marker_test', marker_program, upload_ch, upload_mk, upload_vt)

    try:
        while True:
            for program in channel_group.programs:
                print(f'playing {program}')
                channel_group.arm(program)
                channel_group.run_current_program()
                while not channel_group.was_current_program_finished():
                    print(f'waiting for {program} to finish')
                    time.sleep(1e-2)
    finally:
        channel_group.enable(False)


if __name__ == "__main__":
    import sys
    args = argparse.ArgumentParser('Upload an example pulse to a HDAWG')
    args.add_argument('device_serial', help='device serial of the form dev1234')
    args.add_argument('device_interface', help='device interface', choices=['USB', '1GbE'], default='1GbE', nargs='?')
    args.add_argument('--channels', help='channels to use', choices=range(8), default=[0, 1], type=int, nargs='+')
    args.add_argument('--markers', help='markers to use', choices=range(8*2), default=[], type=int, nargs='*')
    parsed = vars(args.parse_args())

    channels = parsed.pop('channels')
    markers = [(m // 2, m % 2) for m in parsed.pop('markers')]

    logging.basicConfig(stream=sys.stdout)
    example_upload(hdawg_kwargs=parsed, channels=channels, markers=markers)
