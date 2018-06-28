from typing import Dict, Optional, Sequence, Collection, Any, TypeVar, Iterator, Generic
from collections import ChainMap
from collections.abc import Mapping

from qctoolkit.pulses.pulse_template import PulseTemplate
from qctoolkit.pulses.parameters import Parameter

ParameterDict = Dict[str, Parameter]
ParameterEncyclopedia = Dict[str, ParameterDict]

KT, VT = TypeVar('KT'), TypeVar('VT')


class ReadOnlyChainMap(Mapping, Generic[KT, VT]):

    def __init__(self, chain_map: ChainMap) -> None:
        self._chain_map = chain_map

    def __getitem__(self, item: KT) -> VT:
        return self._chain_map[item]

    def __len__(self) -> int:
        return len(self._chain_map)

    def __iter__(self) -> Iterator:
         return iter(self._chain_map)

    def __str__(self) -> str:
        return "ReadOnly{}".format(str(self._chain_map))

    def __repr__(self) -> str:
        return "ReadOnly{}".format(repr(self._chain_map))


class ParameterLibrary:
    """Composes pulse-specific parameter dictionaries from hierarchical source dictionaries.

    Nomenclature: In the following,
    - a ParameterDict refers to a simple dictionary mapping of parameter names to values
    - a ParameterEncyclopedia refers to a dictionary with pulse identifier are keys and ParameterDicts are values.
        Additionally, there might be a 'global' key also referring to a ParameterDict. Parameter values under the 'global'
        key are applied to all pulses.

    Example for a ParameterDict:
    book = dict(foo_param=17.24, bar_param=-2363.4)

    Example for a ParameterEncyclopedia:
    encl = dict(
        global=dict(foo_param=17.24, bar_param=-2363.4),
        test_pulse_1=dict(foo_param=5.125, another_param=13.37)
    )

    ParameterLibrary gets a sequence of ParameterEncyclopedias on initialization. This sequence is
    understood to be in hierarchical order where parameter values in ParameterDicts later in the sequence supersede
    values for the same parameter in earlier dictionaries. In that sense, the order can be understood of being in
    increasing specialization, i.e., most general "default" values come in early ParameterDicts while more
    specialized parameter values (for a specific experiment, hardware setup) are placed in ParameterDicts later in the
    sequence and replace the default values.

    Within a single ParameterDict, parameter values under the 'global' key are applied to every pulse first. If a pulse
    has an identifier for which a key is present in the ParameterDict, the contained parameter values are applied after
    the globals and replace global values (if colliding).
    """

    def __init__(self, parameter_source_dicts: Sequence[ParameterEncyclopedia]) -> None:
        """Creates a ParameterLibrary instance.

        Args:
            parameter_source_dicts (Sequence(Dict(str -> Dict(str -> Parameter)))): A sequence of parameter source dictionaries.
        """
        self._parameter_sources = parameter_source_dicts

    def get_parameters(self, pulse: PulseTemplate, subst_params: Optional[ParameterDict]=None) -> ParameterDict:
        """Returns a dictionary with parameters from the library for a given pulse template.

        The parameter source dictionarys (i.e. the library) given to the ParameterLibrary instance on construction
        are processed to extract the most specialized parameter values for the given pulse as described in the
        class docstring (of ParameterLibrary).

        Additionally, the optional argument subst_params is applied after processing the library to allow for a final
        specialization by custom replacements. subst_params must be a simple parameter dictionary of the form
        parameter_name -> parameter_value .

        Args:
            pulse (PulseTemplate): The PulseTemplate to fetch parameters for.
            subst_params (Dict(str -> Parameter)): An optional additional parameter specialization dictionary to be applied
                after processing the parameter library.
        Returns:
            a mapping giving the most specialized parameter values for the given pulse template. also contains all
            globally specified parameters, even if they are not required by the pulse template.
        """
        maps = []
        if subst_params:
            maps.append(subst_params)
        for param_encl in reversed(self._parameter_sources):
            if pulse.identifier and pulse.identifier in param_encl:
                maps.append(param_encl[pulse.identifier])
            if 'global' in param_encl:
                maps.append(param_encl['global'])
        return ReadOnlyChainMap(ChainMap(*maps))
