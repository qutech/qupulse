"""STANDARD LIBRARY IMPORTS"""
import logging
"""RELATED THIRD PARTY IMPORTS"""

"""LOCAL IMPORTS"""
from PulseTemplate import PulseTemplate

logger = logging.getLogger(__name__)

class BranchPulseTemplate(PulseTemplate):
    """!@brief Conditional branching in a pulse.
    
    A BranchPulseTemplate is a PulseTemplate
    with different structures depending on a certain condition.
    It defines refers to an if-branch and an else-branch, which
    are both PulseTemplates.
    When instantiating a pulse from a BranchPulseTemplate,
    both branches refer to concrete pulses. If the given
    condition evaluates to true at the time the pulse is executed,
    the if-branch, otherwise the else-branch, is chosen for execution.
    This allows for alternative execution 'paths' in pulses.
    
    Both branches must be of the same length.
    """
    def __init__(self):
        super().__init__()
        self.else_branch = None
        self.if_branch = None
        self.condition = None

