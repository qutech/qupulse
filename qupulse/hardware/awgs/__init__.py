# SPDX-FileCopyrightText: 2014-2024 Quantum Technology Group and Chair of Software Engineering, RWTH Aachen University
#
# SPDX-License-Identifier: GPL-3.0-or-later

import lazy_loader as lazy


__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules={'base'},
    submod_attrs={
        'tabor': ['TaborAWGRepresentation', 'TaborChannelPair'],
        'tektronix': ['TektronixAWG'],
        'zihdawg': ['HDAWGRepresentation', 'HDAWGChannelGroup'],
    }
)
