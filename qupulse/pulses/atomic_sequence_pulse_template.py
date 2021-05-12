"""This module defines the AtomicSequencePulseTemplate

The AtomicSequencePulseTemplate is similar to the SequencePulseTemplate, but acts as an atomic template.
This allows definitions of pulses that are combined of pulses that cannot be played by a backend 
(e.g. because of restrictions on the number of samples)

Classes:
    - AtomicSequencePulseTemplate: Defines a pulse
"""

import unittest
import functools
import logging
import numbers
import warnings
from functools import wraps
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np

from qupulse._program.waveforms import SequenceWaveform
from qupulse.expressions import Expression, ExpressionScalar, Number
from qupulse.parameter_scope import DictScope, Scope
from qupulse.pulses.constant_pulse_template import ConstantPulseTemplate
from qupulse.pulses.measurement import (MeasurementDeclaration,
                                        MeasurementDefiner)
from qupulse.pulses.parameters import ParameterNotProvidedException
from qupulse.pulses.pulse_template import (AtomicPulseTemplate, ChannelID,
                                           Loop, MeasurementDeclaration,
                                           PulseTemplate, Transformation,
                                           TransformingWaveform)
from qupulse.pulses.sequence_pulse_template import SequencePulseTemplate
from qupulse.utils.types import FrozenDict, MeasurementWindow

__all__ = ["AtomicSequencePulseTemplate"]


class ignore_warning_decorator:
    def __init__(self, message: str, **kwargs: Any):
        """ Ignore warnings messages using a decorator """
        self.warning_message = message
        self.__dict__.update(kwargs)

    def __call__(self, method: Any) -> Any:
        @wraps(method)
        def wrapper(*args: Any, **kw: Any) -> Any:

            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', self.warning_message)
                print(f'ignore: {self.warning_message}')
                result = method(*args, **kw)
            return result
        return wrapper


def flatten_sets(x: Sequence[Set[Any]]) -> Set[Any]:
    if len(x) == 0:
        return set()
    return set.union(*x)


class AtomicSequencePulseTemplate(AtomicPulseTemplate):  # type: ignore

    def __init__(self, *subtemplates: Union[PulseTemplate],
                 identifier: Optional[str] = None,
                 name: Optional[str] = None,
                 measurements: Optional[List[MeasurementDeclaration]] = None) -> None:
        """ An atomic qupulse template combined multiple templates

        Modelled after SequencePulseTemplate

        Args:
            subtemplates: The list of subtemplates of this
                SequencePulseTemplate as tuples of the form (PulseTemplate, Dict(str -> str)).
            identifier (str): A unique identifier for use in serialization. (optional)

        """
        super().__init__(identifier=identifier, measurements=measurements)

        if len(subtemplates) == 0:
            raise Exception(f'{self.__class_} not valid for empty list of pulse templates')

        self._subtemplates = subtemplates

        if name is None:
            name = 'AtomicSequencePulseTemplate'
        self._name = name

    def __str__(self) -> str:
        return '<{} at %x{}: {}>'.format(self.__class__.__name__, '%x' % id(self), self._name)

    def build_sequence(self) -> None:
        return

    def _as_expression(self) -> Dict[ChannelID, ExpressionScalar]:
        raise NotImplementedError('expression not available for AtomicSequencePulseTemplate')

    @property
    def integral(self) -> Dict[ChannelID, ExpressionScalar]:
        expressions = {channel: 0 for channel in self.defined_channels}

        def add_dicts(x, y):
            return {k: x[k] + y[k] for k in x}

        return functools.reduce(add_dicts, [sub.integral for sub in self._subtemplates], expressions)

    @property
    def parameter_names(self) -> Set[str]:
        """The set of names of parameters required to instantiate this PulseTemplate."""
        x = [p.parameter_names for p in self._subtemplates]
        return flatten_sets(x)

    @property
    def measurement_names(self) -> Set[str]:
        """The set of measurement identifiers in this pulse template."""
        result = set.union(MeasurementDefiner.measurement_names.fget(self),
                           *(st.measurement_names for st in self._subtemplates))
        return result

    @property
    def is_interruptable(self) -> bool:
        """Return true, if this PulseTemplate contains points at which it can halt if interrupted.
        """
        return False

    @property
    def duration(self) -> ExpressionScalar:
        """An expression for the duration of this PulseTemplate."""
        return np.sum([p.duration for p in self._subtemplates])

    @property
    def defined_channels(self) -> Set['ChannelID']:
        """Returns the number of hardware output channels this PulseTemplate defines."""
        dc = [p.defined_channels for p in self._subtemplates]
        return flatten_sets(dc)

    def requires_stop(self) -> bool:  # from SequencingElement
        return False

    def _internal_create_program(self, *,
                                 scope: Scope,
                                 measurement_mapping: Dict[str, Optional[str]],
                                 channel_mapping: Dict[ChannelID, Optional[ChannelID]],
                                 global_transformation: Optional[Transformation],
                                 to_single_waveform: Set[Union[str, PulseTemplate]],
                                 parent_loop: Loop) -> None:
        try:
            measurement_parameters = {parameter_name: scope[parameter_name].get_value()
                                      for parameter_name in self.measurement_parameters}
            parameters = {parameter_name: scope[parameter_name].get_value()
                          for parameter_name in self.parameter_names}
        except KeyError as e:
            raise ParameterNotProvidedException(str(e)) from e

        waveform = self.build_waveform(parameters=parameters,
                                       channel_mapping=channel_mapping)
        if waveform:
            measurements = self.get_measurement_windows(
                parameters=measurement_parameters, measurement_mapping=measurement_mapping)

            if measurements:
                parent_loop.add_measurements(measurements)

            if global_transformation:
                waveform = TransformingWaveform(waveform, global_transformation)

            # make sure all values in the parameters dict are numbers
            if isinstance(parameters, Scope):
                scope = parameters
            else:
                parameters = dict(parameters)
                for parameter_name, value in parameters.items():
                    if isinstance(value, Parameter):
                        parameters[parameter_name] = value.get_value()
                    elif not isinstance(value, Number):
                        parameters[parameter_name] = Expression(value).evaluate_numeric()

                scope = DictScope(values=FrozenDict(parameters), volatile=set())

            # go through children to get measurement
            dummy_loop = Loop()
            for subtemplate in self._subtemplates:
                subtemplate._create_program(scope=scope,
                                            measurement_mapping=measurement_mapping,
                                            channel_mapping=channel_mapping,
                                            global_transformation=global_transformation,
                                            to_single_waveform=to_single_waveform,
                                            parent_loop=dummy_loop)

            sub_measurements = dummy_loop.get_measurement_windows()

            def convert_measurement_format(sub_measurements: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> List[MeasurementWindow]:
                r = []
                for key, item in sub_measurements.items():
                    for idx, start in enumerate(item[0]):
                        r.append((key, start, item[1][idx]))
                return r
            parent_loop.add_measurements(convert_measurement_format(sub_measurements))

            parent_loop.append_child(waveform=waveform)

    def build_waveform(self,
                       parameters: Dict[str, numbers.Real],
                       channel_mapping: Dict[ChannelID, Optional[ChannelID]]) -> SequenceWaveform:
        logging.debug(
            f'{self}.build_waveform: channel_mapping {channel_mapping}, defined_channels {self.defined_channels}')
        if all(channel_mapping[channel] is None
               for channel in self.defined_channels):
            return None

        if self.duration.evaluate_numeric(**parameters) == 0:
            return None

        sub_waveforms = [p.build_waveform(parameters, channel_mapping) for p in self._subtemplates]

        return SequenceWaveform(sub_waveforms)


class TestAtomicSequencePulseTemplate(unittest.TestCase):

    def test_AtomicSequencePulseTemplate_integral(self):
        p1 = ConstantPulseTemplate(10, {'P1': 1.0})
        p2 = ConstantPulseTemplate(10, {'P1': 2.0})
        pt = AtomicSequencePulseTemplate(p1, p2,)
        self.assertDictEqual(pt.integral, {'P1': 30.})

    def test_AtomicSequencePulseTemplate_measurements(self):

        p1 = ConstantPulseTemplate(10, {'P1': 1.0}, measurements=[('a', 0, 2)])
        p2 = ConstantPulseTemplate(10, {'P1': 2.0}, measurements=[('a', 0, 2), ('b', 0, 5)])

        pt = SequencePulseTemplate(p1, p2, measurements=[('c', 0, 3)])

        measurement_windows0 = pt.create_program().get_measurement_windows()

        pt = AtomicSequencePulseTemplate(p1, p2, measurements=[('c', 0, 3)])
        measurement_windows = pt.create_program().get_measurement_windows()

        self.assertEqual(measurement_windows0.keys(), measurement_windows.keys())
        for key in measurement_windows0:
            np.testing.assert_array_equal(measurement_windows0[key], measurement_windows[key])

    def test_AtomicSequencePulseTemplate_measurement_names(self):
        p1 = ConstantPulseTemplate(10, {'P1': 1.0})
        p2 = ConstantPulseTemplate(10, {'P1': 2.0})
        pt = AtomicSequencePulseTemplate(p1, p2,)
        self.assertEqual(pt.measurement_names, set())
        pt = AtomicSequencePulseTemplate(p1, p2, measurements=[('c', 0, 3)])
        self.assertEqual(pt.measurement_names, {'c'})


if __name__ == '__main__':
    import unittest
    unittest.main()
