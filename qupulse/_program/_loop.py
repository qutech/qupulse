# SPDX-FileCopyrightText: 2014-2024 Quantum Technology Group and Chair of Software Engineering, RWTH Aachen University
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Backwards compatibility link to qupulse.program.loop"""

from qupulse.program.loop import *

import qupulse.program.loop

__all__ = qupulse.program.loop.__all__

del qupulse
