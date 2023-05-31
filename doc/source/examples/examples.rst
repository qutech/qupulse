.. _examples:

Examples
========

All examples are provided as static text in this documentation and, additionally, as interactive jupyter notebooks accessible by running ``jupyter notebook`` in the ``/doc/source/examples`` directory of the source tree.


.. toctree::
    :caption: Pulse template types
    :name: pt_types

    00SimpleTablePulse
    00AdvancedTablePulse
    00FunctionPulse
    00PointPulse
    00ComposedPulses
    00ConstantPulseTemplate
    00MultiChannelTemplates
    00MappingTemplate
    00AbstractPulseTemplate
    00ArithmeticWithPulseTemplates
    00RetrospectiveConstantChannelAddition
    00TimeReversal

.. toctree::
    :caption: Pulse template features
    :name: pt_feat

    01PulseStorage
    01Measurements
    01ParameterConstraints

.. toctree::
    :caption: Physically motivated examples
    :name: physical_examples

    03SnakeChargeScan
    03FreeInductionDecayExample
    03GateConfigurationExample
    03DynamicNuclearPolarisation

.. toctree::
    :caption: Pulse playback related examples
    :name: hardware_examples

    02CreatePrograms

The ``/doc/source/examples`` directory also contains some outdated examples for features and functionality that has been changed. These examples start with an underscore i.e. ``_*.ipynb`` and are currently left only for reference purposes.
If you are just learning how to get around in qupulse please ignore them.