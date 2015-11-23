.. _pulsetemplates:

Pulse Templates
---------------

The qctoolkit represents pulses as abstract pulse templates. A pulse template can be understood as a class of pulses that share a similar structure but differ in the concrete amplitude or duration of voltage levels. To this end, pulse templates are parametrizable. Pulse templates are also designed to feature easy reusability of existing templates and conditional execution based on hardware triggers, if supported by the devices.

There are 6 types of different pulse template classes, briefly explained in the following. :class:`.TablePulseTemplate` and :class:`.FunctionPulseTemplate` are used to define the atomic building blocks of pulses in the following ways: :class:`.TablePulseTemplate` allows the user to specify pairs of time and voltage values and choose an interpolation strategy between neighbouring points. :class:`.FunctionPulseTemplate` will accept any mathematical function that maps time to voltage values. All other pulse template variants are then used to construct arbitrarily complex pulses by combining existing ones into new structures: :class:`.SequencePulseTemplate` enables the user to specify a sequence of existing pulse templates (subtemplates) and modify parameter values using a mapping function. :class:`.RepetitionPulseTemplate` is used to simply repeat one existing pulse template a given (constant) number of times. :class:`.BranchPulseTemplate` and :class:`.LoopPulseTemplate` implement conditional execution if supported. All of these pulse template variants can be similarly accessed through the common interface declared by :class:`.PulseTemplate`. [#tree]_ [#pattern]_

Each pulse template can be stored persistently in a human-readable JSON file. :ref:`Read more about serialization <serialization>`.

Parameters
^^^^^^^^^^

As mentioned above, all pulse templates may contain parameters. :class:`.TablePulseTemplate` allows parameter references as table entries on the time and voltage domain. These are represented as :class:`.ParameterDeclaration` objects which are identified by a unique name and can impose lower and upper boundaries to the expected parameter value as well as a default value. :class:`.SequencePulseTemplate` allows to specify a set of new parameter declarations and a mapping of these to the parameter declarations of its subtemplates. This allows renaming of parameters, e.g., to avoid name clashes if several subtemplates declare similarly named parameters. The mapping also allows mathematical transformation of parameter values, such that values that are passed to subtemplates can be obtained by deriving them from one or more other parameter values passed to the :class:`.SequencePulseTemplate`. :class:`.RepetitionPulseTemplate`, :class:`.LoopPulseTemplate` and :class:`BranchPulseTemplate` will simply pass parameters to their subtemplates without modifying them in any way.

The mathematical expressions (for parameter transformation or as the function of the :class:`.FunctionPulseTemplate`) are encapsulated into an :class:`.Expression` class which wraps existing python packages that are able to parse and evaluate expressions given as strings such as `py_expression_eval <https://github.com/AxiaCore/py-expression-eval>`_ and `numexpr <https://github.com/pydata/numexpr>`_.

Obtaining a Concrete Pulse
^^^^^^^^^^^^^^^^^^^^^^^^^^

To obtain a pulse ready for execution on the hardware from a pulse template, the user has to specify parameter values (if parameters were used in the pulse templates in question). In the simplest case, parameters are constant values that can be provided as plain float values. Other cases may require parameter values to be computed based on some measurement values obtained during preceding executions. If so, a subclass of the :class:`.Parameter` class which performs this computations when queried for a value can be provided. To translates the object structures that encode the pulse template in the software into a sequential representation of the concrete pulse with the given parameter values that is understandable by the hardware, the sequencing process has to be invoked. During this process, all parameter values are checked for consistency with the boundaries declared by the parameter declarations and the process is aborted if any violation occurs. :ref:`Read more about the sequencing process <sequencing>`.

.. rubric:: Footnotes
.. [#tree] Regarded as objects in the programming language, each pulse template is a tree of PulseTemplate objects, where the atomic templates (:class:`.TablePulseTemplate` and :class:`.FunctionPulseTemplate` objects) are the leafs while the remaining ones form the inner nodes of the tree.
.. [#pattern] The design of the pulse template class hierarchy is an application of the `Composite Pattern <https://en.wikipedia.org/wiki/Composite_pattern>`_.