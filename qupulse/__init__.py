"""A Quantum compUting PULse parametrization and SEquencing framework."""

import lazy_loader as lazy

__version__ = '0.7'

__getattr__, __dir__, __all__ = lazy.attach_stub(__name__, __file__)

# we explicitly import qupulse to register all deserialization handles
from qupulse import pulses
