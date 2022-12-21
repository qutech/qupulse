import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules={'alazar2'},
    submod_attrs={
        'dac_base': ['DAC'],
        'alazar': ['AlazarCard'],
    }
)
