Creating a New Pulse Template
=============================

This example goes through the workflow of creating a new type of pulse from scratch using the `TablePulseTemplate` class.

.. figure:: _static/example_pulse.*
    :width: 12cm
    :align: center
    :alt: Example pulse

    Figure 1: Example pulse shape. The crosses mark the points that go into the table, they are connected using different
    interpolation strategies.

`Figure 1` shows the pulse shape we want to build in this tutorial. The crosses mark the supporting points that are
connected using different interpolation strategies. We want to parametrize the position of the supporting points using
the time at which the voltage rises (t\ :sub:`a` ) as *start* and the length of the rise afterwards as *length*.

We start with an empty `TablePulseTemplate` object and add supporting points::

    squarePulse = TablePulseTemplate() # Prepare new empty Pulse
    # Then add pulses sequentially
    squarePulse.add_entry(0, 0)
    squarePulse.add_entry('ta', 'va', interpolation='hold') # hold is the standard interpolation value
    squarePulse.add_entry('tb', 'vb', interpolation='linear')
    squarePulse.add_entry('end', 0, interpolation='invhold') # TODO: replace 'invhold' with a better name

The supporting points must be added in order of increasing time. The next step is adding a `Mapping` to calculate the
values for `ta`, `tb`, `va`, `vb` and `end`.