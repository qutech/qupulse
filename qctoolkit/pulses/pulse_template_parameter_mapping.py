from qctoolkit.pulses.mapping_pulse_template import MappingPulseTemplate

__all__ = ["MappingPulseTemplate"]

import warnings
warnings.warn("MappingPulseTemplate was moved from qctoolkit.pulses.pulse_template_parameter_mapping to "
              "qctoolkit.pulses.mapping_pulse_template. Please consider fixing your stored pulse templates by loading "
              "and storing them anew.", DeprecationWarning)
