import lazy_loader as lazy


__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules={'base'},
    submod_attrs={
        'tabor': ['TaborAWGRepresentation', 'TaborChannelPair'],
        'tektronix': ['TektronixAWG'],
        'zihdawg': ['HDAWGRepresentation', 'HDAWGChannelGroup'],
    }
)
