# SPDX-FileCopyrightText: 2014-2024 Quantum Technology Group and Chair of Software Engineering, RWTH Aachen University
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""..deprecated:: 0.1
"""

from qupulse.pulses.mapping_pulse_template import MappingPulseTemplate

__all__ = ["MappingPulseTemplate"]

import warnings
warnings.warn("MappingPulseTemplate was moved from qupulse.pulses.pulse_template_parameter_mapping to "
              "qupulse.pulses.mapping_pulse_template. Please consider fixing your stored pulse templates by loading "
              "and storing them anew.", DeprecationWarning)

from qupulse.serialization import SerializableMeta
SerializableMeta.deserialization_callbacks["qupulse.pulses.pulse_template_parameter_mapping.MappingPulseTemplate"] = SerializableMeta.deserialization_callbacks[MappingPulseTemplate.get_type_identifier()]
