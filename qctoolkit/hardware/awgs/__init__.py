__all__ = []

try:
    from qctoolkit.hardware.awgs.tabor import TaborAWGRepresentation, TaborChannelPair
    __all__.extend(["TaborAWGRepresentation", "TaborChannelPair"])
except ImportError:
    pass
