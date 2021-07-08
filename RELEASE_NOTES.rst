

.. towncrier release notes start

qupulse 0.5.1 (2021-07-08)
==========================

Features
--------

- Add `evaluate_with_exact_rationals` method to `ExpressionScalar` (`#546 <https://github.com/qutech/qupulse/issues/546>`_)
- New feature based AWG abstraction. Can be found in `qupulse.hardware.feature_awg`. Allows easier code reuse across awg drivers. (`#557 <https://github.com/qutech/qupulse/issues/557>`_)
- Add ConstantPulseTemplate (`#565 <https://github.com/qutech/qupulse/issues/565>`_)
- Add interface to use `atsaverage` auto rearm (`#566 <https://github.com/qutech/qupulse/issues/566>`_)
- Adds the methods `is_constant`, `constant_value_dict` and `constant_value` to Waveform class to allow more efficient AWG usage. (`#588 <https://github.com/qutech/qupulse/issues/588>`_)


Bugfixes
--------

- Fix TimeType comparisons with non-finite floats (inf, -inf, NaN) (`#536 <https://github.com/qutech/qupulse/issues/536>`_)
- Improve alazar usability:
    - Do not touch the default config when arming a measurement
    - Keep current config in a seperate field
    - Extend record to a multiple of a configurable value (4KB by default) (`#571 <https://github.com/qutech/qupulse/issues/571>`_)
- Replace pytabor and teawg with tabor_control to support newer(>=1.11) pyvisa versions (`#599 <https://github.com/qutech/qupulse/issues/599>`_)
- Fix `repr` of `ExpressionScalar` when constructed from a sympy expression. Also replace `Expression` with `ExpressionScalar` in `repr`. (`#604 <https://github.com/qutech/qupulse/issues/604>`_)


Deprecations and Removals
-------------------------

- Deprecate HashableNumpyArray due to its inconsistency. (`#408 <https://github.com/qutech/qupulse/issues/408>`_)
- Drop support for python 3.5 (`#504 <https://github.com/qutech/qupulse/issues/504>`_)
- Remove deprecated `external_parameters` keyword argument from SequencePT and AtomicMultiChannelPT (`#592 <https://github.com/qutech/qupulse/issues/592>`_)
- Deprecate boolean `duration` argument of `AtomicMultiChannelPulseTemplate` and remove duration check in `__init__`. (`#593 <https://github.com/qutech/qupulse/issues/593>`_)


0.5.1
=====

- General:
   - Unify `TimeType.from_float` between fractions and gmpy2 backend behaviour (fixes issue 529).

0.5
=====

- General:
   - Improve `TimeType` consistency by leveraging str(float) for rounding by default.
   - Add support for sympy==1.5
   - Add volatile parameters. Repetition counts can now be changed at runtime in some cases (useful for DNP). See `volatile` kwarg of `create_program`

- Hardware:
   - Add a `measure_program` method to the DAC interface. This method is used by the QCoDeS integration.
   - Add a `set_measurement_mask` to DAC interface. This method is used by the QCoDeS integration.
   - Add a `get_sample_times` util method to share code for exact and fast sample time calculation
   - Add a driver for Tektronix AWG5000/7000
   - Add a driver for Zurich Instruments HDAWG
   - Warn the user if waveforms need to be concatenated to be compatible with hardware requirements.

- Pulse Templates:
    - Add `__repr__` and `__format__` for easier inspection
    - `MappingPulseTemplate`:
        - `allow_partial_parameter_mapping` is now True as a default. The default can be changed with the class variable `MappingPulseTemplate.ALLOW_PARTIAL_PARAMETER_MAPPING`.
        - Add specializations for `map_parameters` because the auto-inference of the return type did not work for empty input.
        - Channels mapped to None are now dropped
    - Add simple arithmetic operations for pulse templates
        - offset and scaling with scalars
        - addition with atomic pulse templates

- Expressions:
    - Expressions can now be formatted as floats if they do not have free variables

- Parameters:
    - Replace Parameter class with Scope
    - Parameter class is now deprecated

- Backward incompatible changes:
    - Removed deprecated classes:
      - Sequencer: Replaced by PulseTemplate.create_program method
      - Condition: Never used
      - InstructionBlock: Old representation of programs. Replaced by Loop
      - MultiChannelProgram: Was required in the instruction block framework

0.4
=====

- General:
    - Add utility function `qupulse.utils.types.has_type_interface` and use it to circumvent autoreload triggered isinstance fails
    - Add utility function `qupulse.utils.time_from_fraction` to make creation from numerator and denominator obvious.

- Pulse Templates:
    - `MappingPulseTemplate`:
        - Raise a ValueError if more than one inner channel is mapped to the same outer channel
    - Plotting:
        - Make `plotting.render` behaviour and return value consistent between calls with `InstructionBlock` and `Loop`. Render now always returns 3 arguments.

0.3
=====

- General:
    - Introduce qupulse.utils.isclose (an alias for math.isclose if available)
    - Dropped support for Python 3.4 in setup.py due to incompatible syntax in qupulse.
    - Official support for Python 3.7 has begun.

- Pulse Templates:
    - `AtomicMultichannelPulseTemplate`:
        - Add duration keyword argument & example (see MultiChannelTemplates notebook)
        - Make duration equality check approximate (numeric tolerance)
    - Plotting:
        - Add `time_slice` keyword argument to render() and plot()
    - Add `AbstractPulseTemplate` class
    - `PointPulseTemplate`:
        - Fixed bug in integral evaluation
    - Add `ParallelConstantChannelPulseTemplate` which allows adding a constant valued channel to an arbitrary pulse template

- Expressions:
    - Make ExpressionScalar hashable
    - Fix bug that prevented evaluation of expressions containing some special functions (`erfc`, `factorial`, etc.)

- Parameters:
    - `ConstantParameter` now accepts a `Expression` without free variables as value (given as `Expression` or string)

0.2
=====

- General:
   - officially removed support for Python 3.3 (qupulse and dependencies are not compatible anymore)

- Serialization / Storage:
   - Added functionality to easily access available content/identifiers in `PulseStorage` and `StorageBackend`.
   - DEPRECATED `list_contents()` of `StorageBackend` (use `contents property` instead).
   - DEPRECATED: `CachingBackend` because its functionality is a subset of `PulseStorage`.

- Expressions:
   - Fixed bug in `Expression.evaluate_numeric` if result is array of numeric sympy objects
