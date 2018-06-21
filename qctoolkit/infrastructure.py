from typing import Dict, Optional, Sequence, Collection, Any

from qctoolkit.pulses.pulse_template import PulseTemplate
from qctoolkit.pulses.parameters import Parameter

ParameterDict = Dict[str, Parameter]
ParameterEncyclopedia = Dict[str, ParameterDict]


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

    @staticmethod
    def _filter_dict(dictionary: Dict[str, Any], filter: Collection[str]) -> Dict[str, Any]:
        return {k: dictionary[k] for k in dictionary if k in filter}

    @staticmethod
    def _update_params_dict(params: Dict[str, Any],
                            new_params_source: Dict[str, Any],
                            pulse_parameter_names: Collection[str]):
        params.update(ParameterLibrary._filter_dict(new_params_source, pulse_parameter_names))

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
        """
        params = dict()
        parameter_names = set(pulse.parameter_names) # paranoid. pulse.parameter_names is not currently guaranteed to be a set......
        for param_level_dict in self._parameter_sources:
            if 'global' in param_level_dict:
                self._update_params_dict(params, param_level_dict['global'], parameter_names)
            if pulse.identifier and pulse.identifier in param_level_dict:
                self._update_params_dict(params, param_level_dict[pulse.identifier], parameter_names)
        if subst_params:
            self._update_params_dict(params, subst_params, parameter_names)
        return params
