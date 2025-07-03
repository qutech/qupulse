.. _pulsetemplates:

Pulse Templates
---------------

qupulse represents pulses as abstract pulse templates. A pulse template can be understood as a class of pulses that share a similar structure but differ in the concrete amplitude or duration of voltage levels or repetition counts. To this end, pulse templates are parameterizable. Pulse templates are also designed to feature easy reusability of existing templates. The process of plugging in values for a pulse templates parameters is called instantiation. This is done via :meth:`~.PulseTemplate.create_program`. The result is a deterministic description of all voltages at all times and is called a program.

There are multiple types of different pulse template classes, briefly explained in the following.

:class:`.TablePulseTemplate`, :class:`.PointPulseTemplate` and :class:`.FunctionPulseTemplate` are used to define the atomic building blocks of pulses in the following ways: :class:`.TablePulseTemplate` and :class:`.PointPulseTemplate` allow the user to specify pairs of time and voltage values and choose an interpolation strategy between neighbouring points. Both templates support multiple channels but :class:`.TablePulseTemplate` allows for different time values for different channels meaning that the channels can change their voltages at different times. :class:`.PointPulseTemplate` restricts this to switches at the same time by interpreting the voltage as a vector and provides a more convenient interface for this case.
:class:`.FunctionPulseTemplate` accepts any mathematical function that maps time to voltage values. Internally it uses :class:`qupulse.expressions.Expression` for function evaluation.

All other pulse template classes are then used to construct arbitrarily complex pulse templates by combining existing ones into new structures [#tree]_:
:class:`.SequencePulseTemplate` enables the user to specify a sequence of existing pulse templates (subtemplates) and modify parameter values using a mapping function.
:class:`.RepetitionPulseTemplate` is used to simply repeat one existing pulse template a given number of times.
:class:`.ForLoopPulseTemplate` is similar but allows a parametrization of the loop body with the loop index.
One special pulse template is the :class:`.mapping_pulse_template.MappingPulseTemplate` which allows the renaming of channels and measurements as well as mapping parameters by mathematical expressions.

In some cases, it is desired to write a pulse which partly consists of placeholder pulses. For this the :class:`.AbstractPulseTemplate` was included.

You can do some simple arithmetic with pulses which is implemented via :class:`.ArithmeticPulseTemplate` and :class:`.ArithmeticAtomicPulseTemplate`. The relevant arithmetic operators are overloaded so you do not need to use these classes directly.

All of these pulse template variants can be similarly accessed through the common interface declared by the :class:`.PulseTemplate` base class. [#pattern]_

As the class names are quite long the recommended way for abbreviation is to use the aliases defined in :py:mod:`~qupulse.pulses`. For example :class:`.FunctionPulseTemplate` is aliased as :class:`.FunctionPT`

Each pulse template can be stored persistently in a human-readable JSON file. :ref:`Read more about serialization <serialization>`.

Parameters
^^^^^^^^^^

As mentioned above, all pulse templates may depend on parameters. During pulse template initialization the parameters simply are the free variables of expressions that occur in the pulse template. For example the :class:`.FunctionPulseTemplate` has expressions for its duration and the voltage time dependency i.e. the underlying function. Some pulse templates provided means to constrain parameters by accepting a list of :class:`.ParameterConstraint` which encapsulate comparative expressions that must evaluate to true for a given parameter set to successfully instantiate a pulse from the pulse template. This can be used to encode physical or logical parameter boundaries at pulse level.

The mathematical expressions (for parameter transformation or as the function of the :class:`.FunctionPulseTemplate`) are encapsulated into an :class:`.sympy.Expression` class which wraps `sympy <http://www.sympy.org/en/index.html>`_ for string evaluation by default. Other more performant or secure backends can potentially be implemented by conforming to the :class:`.protocol.Expression`.

Parameters can be mapped to arbitrary expressions via :class:`.mapping_pulse_template.MappingPulseTemplate`. One use case can be deriving pulse parameters from physical quantities.

On instantiation the parameters are evaluated via parameter scopes which are defined in :py:mod:`~qupulse.parameter_scope`. This is currently only relevant for internal usage but might be useful if you want to develop a more sophisticated parameter management.

Measurements
^^^^^^^^^^^^

Pulses are usually used to manipulate the state of some physical system and the system's response has to be somehow validated and thus measured. qupulse pulse templates allow to define measurement windows that specify at what times measurements should be made and identify those windows with an identifier.
After the pulse templates are instantiated, uploading the resulting pulses to the hardware setup will cause qupulse to also configure corresponding measurement devices according to the specified measurement windows.

Metadata
^^^^^^^^

Pulse templates have a :attr:`.PulseTemplate.metadata` attribute which is the only mutable part of the class. It is the place for translation hints and user-data. Currently, the only effect on qupulse is that all pulse templates that have the metadata field :attr:`.TemplateMetadata.to_single_waveform` set to 'always' are translated into a single waveform.

Since it is a regular python object, you can freely add any fields you like to it without changing the class. All your custom fields will be serialized by qupulse if you want to save the template to disk.

Due to the non-constant nature of the metadata field it is ignored when comparing pulse templates. This is necessary, because pulse templates implement hash.

Convenience Functionality
^^^^^^^^^^^^^^^^^^^^^^^^^

In addition to the fundamental concepts and subclassing structure, :class:`.PulseTemplate` offers a variety of
**convenience features** that make it easier to combine and modify pulses:

- **Convenience Methods**:

  - :meth:`.PulseTemplate.with_parallel_channels` attaches additional constant channel values in parallel to the
    existing channels of a pulse.
  - :meth:`.PulseTemplate.with_repetition` repeats a pulse a specified number of times.
  - :meth:`.PulseTemplate.with_mapping` enables mapping or renaming of parameters, channel names, and measurement
    identifiers.
  - :meth:`.PulseTemplate.with_iteration` wraps a pulse into a loop construct, allowing iteration over an index with a
    defined range.
  - :meth:`.PulseTemplate.with_time_reversal` generates a time-reversed version of a pulse.
  - :meth:`.PulseTemplate.with_appended` concatenates additional pulse templates, creating a sequence which can be more
    concise than manually building a :class:`.SequencePulseTemplate`.

- **Padding**:

  - :meth:`.PulseTemplate.pad_to` extends the duration of a pulse template to a desired length, automatically appending
    a constant pulse segment that matches the final output values.

- **Properties**:

  - :attr:`.PulseTemplate.duration` provides an expression for the pulse’s total duration.
  - :attr:`.PulseTemplate.initial_values` and :attr:`.PulseTemplate.final_values` specify the voltage levels at the
    start and end of the pulse, respectively.
  - :attr:`.PulseTemplate.integral` returns an expression evaluating the area under the pulse for each channel.

- **Arithmetic**:

  - Basic arithmetic operators (``+``, ``-``, ``*``, ``/``) are overloaded to allow direct numerical combination or
    scaling of pulses.
  - The :meth:`@` operator (matrix multiplication) is repurposed to represent time-wise concatenation of pulses (see
    :meth:`.SequencePulseTemplate.concatenate`).

These convenience features allow for more compact, readable, and flexible pulse definitions, helping to focus on the
overall structure of the experiment.

Obtaining a Concrete Pulse (Pulse Instantiation)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To obtain a pulse ready for execution on the hardware from a pulse template, the user has to specify parameter values (if parameters were used in the pulse templates in question). Parameters can be marked as volatile. This will make the instrument drivers upload the program in such a way that the parameters can easily be updated. This restricts the parameters that can be marked as volatile. qupulse should raise a meaningful exception if the required changes are too complex or not supported on the used instrument.

In order to translate the object structures that encode the pulse template in the software into a (sequential) representation of the concrete pulse with the given parameter values that is understandable by the hardware, we proceed in several steps.

First, the :meth:`.PulseTemplate.create_program` checks parameter consistency with parameter constraints and translates the pulse template into an instantiated program object. The nature of this program depends on the targeted hardware and is determined by the ``program_builder`` keyword argument. This program is further interpreted and sequenced by the hardware backend code (in :py:mod:`~qupulse.hardware`).

See :ref:`instantiating` for a more in-depth explanation of instantiating pulses.

Relevant Examples
^^^^^^^^^^^^^^^^^

Examples demonstrating the construction of pulse templates and parameters from very simple to somewhat more complex pulses are

* :ref:`/examples/00SimpleTablePulse.ipynb`
* :ref:`/examples/00AdvancedTablePulse.ipynb`
* :ref:`/examples/00FunctionPulse.ipynb`
* :ref:`/examples/00PointPulse.ipynb`
* :ref:`/examples/00ComposedPulses.ipynb`
* :ref:`/examples/00ConstantPulseTemplate.ipynb`
* :ref:`/examples/00MappingTemplate.ipynb`
* :ref:`/examples/00MultiChannelTemplates.ipynb`
* :ref:`/examples/00ArithmeticWithPulseTemplates.ipynb`

:ref:`/examples/01ParameterConstraints.ipynb` demonstrates the mentioned parameter constraints.

:ref:`/examples/01Measurements.ipynb` shows how to specify measurements.

:ref:`/examples/02CreatePrograms.ipynb` illustrates usage of the :meth:`.PulseTemplate.create_program` method.

:ref:`physical_examples` show realistic use cases of pulse templates.

.. rubric:: Footnotes
.. [#tree] Regarded as objects in the programming language, each pulse template is a tree of PulseTemplate objects, where the atomic templates (:class:`.TablePulseTemplate` and :class:`.FunctionPulseTemplate` objects) are the leafs while the remaining ones form the inner nodes of the tree.
.. [#pattern] The design of the pulse template class hierarchy is an application of the `Composite Pattern <https://en.wikipedia.org/wiki/Composite_pattern>`_.
