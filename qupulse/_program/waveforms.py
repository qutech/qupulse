# SPDX-FileCopyrightText: 2014-2024 Quantum Technology Group and Chair of Software Engineering, RWTH Aachen University
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Backwards compatibility link to qupulse.program.waveforms"""

from qupulse.program.waveforms import *

import qupulse.program.waveforms

__all__ = qupulse.program.waveforms.__all__

del qupulse
