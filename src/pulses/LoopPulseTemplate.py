"""STANDARD LIBRARY IMPORTS"""
import logging
"""RELATED THIRD PARTY IMPORTS"""

"""LOCAL IMPORTS"""
from PulseTemplate import PulseTemplate

logger = logging.getLogger(__name__)

class LoopPulseTemplate(PulseTemplate):
    """!@brief Conditional looping in a pulse.
    
    A LoopPulseTemplate is a PulseTemplate which is repeated
    during execution as long as a certain condition holds.
    """
    pass