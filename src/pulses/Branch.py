"""STANDARD LIBRARY IMPORTS"""
import logging
"""RELATED THIRD PARTY IMPORTS"""

"""LOCAL IMPORTS"""
from PulseTemplate import PulseTemplate

logger = logging.getLogger(__name__)

class Branch(PulseTemplate):
    """docstring for Branch"""
    def __init__(self):
        super().__init__()
        self.else_branch = None
        self.if_branch = None
        self.condition = None

