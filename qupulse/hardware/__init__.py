# SPDX-FileCopyrightText: 2014-2024 Quantum Technology Group and Chair of Software Engineering, RWTH Aachen University
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Contains drivers for AWG control and digitizer configuration as well as a unifying interface to all instruments:
:class:`~qupulse.hardware.setup.HardwareSetup`"""

from qupulse.hardware.setup import HardwareSetup

from qupulse.hardware import awgs
from qupulse.hardware import dacs

__all__ = ["HardwareSetup", "awgs", "dacs"]
