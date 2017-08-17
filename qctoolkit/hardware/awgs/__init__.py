__all__ = []

try:
    from qctoolkit.hardware.awgs.tabor import TaborAWGRepresentation, TaborChannelPair
    __all__.extend(["TaborAWGRepresentation", "TaborChannelPair"])
except ImportError:
    pass

try:
    from qctoolkit.hardware.awgs.tektronix import TektronixAWG
    __all__.extend("TektronixAWG")
except ImportError:
    pass
