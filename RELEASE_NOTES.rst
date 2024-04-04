

.. towncrier release notes start

qupulse 0.9 (2023-11-08)
========================

Features
--------

- Add `__pow__` as a repetition shortcut. This means you can do `my_pulse_template ** 5` or `my_pulse_template ** 'my_repetition_count'`. (`#692 <https://github.com/qutech/qupulse/issues/692>`_)
- Promote ``qupulse.expression`` to a subpackage and create ``qupulse.expression.protocol`` with protocol classes that define the expression interface that is supposed to be used by qupulse.
  The ```sympy`` based implementation is moved to ``qupulse.expressions.sympy`` and imported in ``qupulse.expressions``.

  The intended use is to be able to use less powerful but faster implementations of the ``Expression`` protocol where appropriate.
  In this first iteration, qupulse still relies on internals of the ``sympy`` based implementation in many places which is to be removed in the future. (`#750 <https://github.com/qutech/qupulse/issues/750>`_)
- Promote parts of the private subpackage `qupulse._program` to the public subpackage `qupulse.program`, i.e. `loop`, `volatile`, `transformation` and `waveforms`. This allows external packages/drivers to rely on stability of the `Loop` class. (`#779 <https://github.com/qutech/qupulse/issues/779>`_)
- Add ``PulseTemplate.pad_to`` method to help padding to minimal lengths or multiples of given durations. (`#801 <https://github.com/qutech/qupulse/issues/801>`_)


Misc
----

- `#771 <https://github.com/qutech/qupulse/issues/771>`_


qupulse 0.8 (2023-03-28)
========================

Features
--------

- New two dimensional plotting function ``qupulse.pulses.plotting.plot_2d``. (`#703 <https://github.com/qutech/qupulse/issues/703>`_)
- Add support for time dependent expressions for arithmetics with atomic pulse templates i.e. ``ParallelChannelPT`` and
  ``ArithmeticPT`` support time dependent expressions if used with atomic pulse templates.
  Rename ``ParallelConstantChannelPT`` to ``ParallelChannelPT`` to reflect this change. (`#709 <https://github.com/qutech/qupulse/issues/709>`_)
- Add ``with_`` family of helper methods to ``PulseTemplate`` to allow convinient and easily discoverable pulse template
  combination. (`#710 <https://github.com/qutech/qupulse/issues/710>`_)
- The plotting module is now located at `qupulse.plotting`. There is a legacy alias at `qupulse.pulses.plotting`. (`#735 <https://github.com/qutech/qupulse/issues/735>`_)


Deprecations and Removals
-------------------------

- Remove the ``Parameter``, ``MappedParameter`` and ``ConstantParameter`` classes that where deprecated in version 0.5. (`#512 <https://github.com/qutech/qupulse/issues/512>`_)
- Drop support for python version 3.7. (`#760 <https://github.com/qutech/qupulse/issues/760>`_)


qupulse 0.7 (2022-10-05)
========================

Features
--------

- Add optional numba uses in some cases. (`#501 <https://github.com/qutech/qupulse/issues/501>`_)
- Add `initial_values` and `final_values` attributes to `PulseTemplate`.

  This allows pulse template construction that depends on features of arbitrary existing pulses i.e. like extension until
  a certain length. (`#549 <https://github.com/qutech/qupulse/issues/549>`_)
- Support sympy 1.9 (`#615 <https://github.com/qutech/qupulse/issues/615>`_)
- Add option to automatically reduce the sample rate of HDAWG playback for piecewise constant pulses.
  Use `qupulse._program.seqc.WaveformPlayback.ENABLE_DYNAMIC_RATE_REDUCTION` to enable it. (`#622 <https://github.com/qutech/qupulse/issues/622>`_)
- Add a TimeReversalPT. (`#635 <https://github.com/qutech/qupulse/issues/635>`_)
- Add specialied parameter Scope for ForLoopPT. This increases performance by roughly a factor of 3 for long ranges! (`#642 <https://github.com/qutech/qupulse/issues/642>`_)
- Add sympy 1.10 support and make `ExpressionVector` hashable. (`#645 <https://github.com/qutech/qupulse/issues/645>`_)
- `Serializable` is now comparable via it's `get_serialized_data`. `PulseTemplate` implements `Hashable` via the same. (`#653 <https://github.com/qutech/qupulse/issues/653>`_)
- Add an interface that uses `atsaverage.config2`. (`#686 <https://github.com/qutech/qupulse/issues/686>`_)


Bugfixes
--------

- `floor` will now return an integer in lambda expressions with numpy to allow usage in ForLoopPT range expression. (`#612 <https://github.com/qutech/qupulse/issues/612>`_)


Deprecations and Removals
-------------------------

- Drop `cached_property` dependency for python>=3.8. (`#638 <https://github.com/qutech/qupulse/issues/638>`_)
- Add frozendict dependency to replace handwritten solution. Not having it installed will break in a future release
  when the old implementation is removed. (`#639 <https://github.com/qutech/qupulse/issues/639>`_)
- Drop python 3.6 support. (`#656 <https://github.com/qutech/qupulse/issues/656>`_)


qupulse 0.6 (2021-07-08)
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
