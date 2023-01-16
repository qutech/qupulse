.. _learners_guide:

Learners Guide - writing pulses with qupulse
--------------------------------------------

This is a little guide through the documentation of qupulse with the idea that *you* as an interested person can find the materials corresponding to the desired skills.

The following steps assume that you have qupulse installed and are able to run the example notebooks.


Basic pulse writing
^^^^^^^^^^^^^^^^^^^

.. topic:: Info

    **Estimated time:**
    30 minutes for reading
    60 minutes for the examples
    60 minutes for experimenting

    **Target group:**

    **Learning Goals:** The learner is able to define a parameterized nested pulse template.

**Learning Task 1:** Read the concept section about :ref:`pulsetemplates`.

**Exercise Task 1:** Go through the following examples that introduce the shipped atomic pulse templates:

* :ref:`/examples/00SimpleTablePulse.ipynb`
* :ref:`/examples/00AdvancedTablePulse.ipynb`
* :ref:`/examples/00FunctionPulse.ipynb`
* :ref:`/examples/00PointPulse.ipynb`
* :ref:`/examples/00ConstantPulseTemplate.ipynb`

**Exercise Task 2:** Go through the following examples that introduce the most important composed pulse templates:

* :ref:`/examples/00ComposedPulses.ipynb`
* :ref:`/examples/00MappingTemplate.ipynb`
* :ref:`/examples/00MultiChannelTemplates.ipynb`

**Exercise Task 3:** Go through the following examples that introduce other useful pulse templates:

* :ref:`/examples/00ArithmeticWithPulseTemplates.ipynb`
* :ref:`/examples/00RetrospectiveConstantChannelAddition.ipynb`
* :ref:`/examples/00TimeReversal.ipynb`

Pulse template features
^^^^^^^^^^^^^^^^^^^^^^^

.. topic:: Info

    **Estimated time:**
    20 minutes for reading
    60 minutes for the examples
    60 minutes for experimenting

    **Target group:**

    **Learning Goals:** The learner to save pulse templates to the file system.
    The learner can use pulse identifiers measurement windows and parameter constraints as needed. The learner is able to verify pulse and measurement windows are as intended for a given parameter set by plotting and inspecting. The learner can load pulses from a file and other valid datasources and use them as a building block in their own pulses.


**Learning Task 2:** Read the concept section about :ref:`serialization`.

**Exercise Task 4:** Go through the :ref:`/examples/01PulseStorage.ipynb` example. It shows how to load and store pulse templates to disk.

**Exercise Task 5:** Go through the :ref:`/examples/01Measurements.ipynb` example. It shows how to define and inspect measurement windows.

**Exercise Task 6:** Go through the :ref:`/examples/01ParameterConstraints.ipynb` example. It shows how to use parameter constraints to enforce invariants.

**Exercise Task 7:** Go through the :ref:`/examples/03SnakeChargeScan.ipynb` example which shows a realistic pulse.

Hardware capabilities and limitations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This section introduces aspects of the hardware that are relevant for every experimenter.

This section is incomplete.

.. topic:: Info

    **Estimated time:**
    20 minutes for reading

    **Target group:** People who want to use qupulse in an experiment.

    **Learning Goals:**
    The learner can identify if a hardware limitation related exception that is raised is due to an error on their end and mitigate it.
    The learner understands capabilities of at least one type of AWGs.


**Learning Task 1:**

Read :ref:`program` and :ref:`awgs`.


Setup an experiment
^^^^^^^^^^^^^^^^^^^

This section is under construction. There is currently an outdated example :ref:`/examples/_HardwareSetup.ipynb`
