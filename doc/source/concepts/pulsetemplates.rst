.. _pulsetemplates:

Pulse Templates
---------------

The qctoolkit represents pulses as abstract pulse templates. A pulse template can be understood as a class of pulses that share a similar structure but differ in the concrete amplitude or duration of voltage levels. To this end, pulse templates are parameterizable. Pulse templates are also designed to feature easy reusability of existing templates and conditional execution based on hardware triggers, if supported by the devices. The process of plugging in values for a pulse templates parameters is called instantiation.

There are multiple types of different pulse template classes, briefly explained in the following. :class:`.TablePulseTemplate`, :class:`.PointPulseTemplate`: and :class:`.FunctionPulseTemplate` are used to define the atomic building blocks of pulses in the following ways: :class:`.TablePulseTemplate` and :class:`.PointPulseTemplate` allow the user to specify pairs of time and voltage values and choose an interpolation strategy between neighbouring points. Both templates support multiple channels but :class:`.TablePulseTemplate` allows for different time values for different channels meaning that the channels can change their voltages at different times. :class:`.PointPulseTemplate` restricts this to switches at the same time by interpreting the voltage as an vector and provides a more convenient interface for this case.
:class:`.FunctionPulseTemplate` accepts any mathematical function that maps time to voltage values. Internally it uses :class:`.Expression` for function evaluation.
All other pulse template variants are then used to construct arbitrarily complex pulses by combining existing ones into new structures: :class:`.SequencePulseTemplate` enables the user to specify a sequence of existing pulse templates (subtemplates) and modify parameter values using a mapping function. :class:`.RepetitionPulseTemplate` is used to simply repeat one existing pulse template a given number of times. :class:`.ForLoopPulseTemplate` is similar but allows a parametrization of the loop body with the loop index. In the future there will be pulse templates that allow conditional execution like :class:`.BranchPulseTemplate` and :class:`.WhileLoopPulseTemplate`. All of these pulse template variants can be similarly accessed through the common interface declared by :class:`.PulseTemplate`. [#tree]_ [#pattern]_ One special pulse template is the :class:`.MappingPulseTemplate` which allows the renaming of channels and measurements as well as mapping parameters by mathematical expressions.

As the class names are quite long the recommended way for abbreviation is to use the aliases defined in :py:mod:`~qctoolkit.pulses`. For example :class:`.FunctionPulseTemplate` is aliased as :class:`.FunctionPT`

Each pulse template can be stored persistently in a human-readable JSON file. :ref:`Read more about serialization <serialization>`.

Parameters
^^^^^^^^^^

As mentioned above, all pulse templates may depend on parameters. During pulse template initialization the parameters simply are the free variables of expressions that occur in the pulse template. For example the :class:`.FunctionPulseTemplate` has expressions for its duration and the voltage time dependency. Some pulse templates provided means to constrain parameters by accepting a list of :class:`.ParameterConstraint` which are expressions which must be true.

The mathematical expressions (for parameter transformation or as the function of the :class:`.FunctionPulseTemplate`) are encapsulated into an :class:`.Expression` class which wraps `sympy <http://www.sympy.org/en/index.html>`_ for string evaluation.

In the future, it will be possible to have parameters dependent on measurement outcomes or other events. This is the reason :class:`.Parameter` objects are passed through on pulse instantiation.

Obtaining a Concrete Pulse
^^^^^^^^^^^^^^^^^^^^^^^^^^

To obtain a pulse ready for execution on the hardware from a pulse template, the user has to specify parameter values (if parameters were used in the pulse templates in question). In the simplest case, parameters are constant values that can be provided as plain float values. Other cases may require parameter values to be computed based on some measurement values obtained during preceding executions. If so, a subclass of the :class:`.Parameter` class which performs this computations when queried for a value can be provided. In order to translate the object structures that encode the pulse template in the software into a sequential representation of the concrete pulse with the given parameter values that is understandable by the hardware, the sequencing process has to be invoked. During this process, all parameter values are checked for consistency with the boundaries declared by the parameter declarations and the process is aborted if any violation occurs. :ref:`Read more about the sequencing process <sequencing>`.

Relevant Examples
^^^^^^^^^^^^^^^^^

Examples demonstrating the use of pulse templates and parameters are :ref:`examples/00SimpleTablePulse.ipynb`, :ref:`examples/01SequencePulse.ipynb` and :ref:`examples/02FunctionPulse.ipynb`.

.. rubric:: Footnotes
.. [#tree] Regarded as objects in the programming language, each pulse template is a tree of PulseTemplate objects, where the atomic templates (:class:`.TablePulseTemplate` and :class:`.FunctionPulseTemplate` objects) are the leafs while the remaining ones form the inner nodes of the tree.
.. [#pattern] The design of the pulse template class hierarchy is an application of the `Composite Pattern <https://en.wikipedia.org/wiki/Composite_pattern>`_.
