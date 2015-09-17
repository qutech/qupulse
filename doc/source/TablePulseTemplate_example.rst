.. _TablePulseTemplate_example:

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
    squarePulse.add_entry('end', 0, interpolation='jump')

The supporting points must be added in order of increasing time. Now we can do two things:
- leave the pulse and instantiate it using concrete values for the parameters, or
- re-parametrize the pulse with more expressive parameters.

::

    #The first one is simple, just set up the parameter dictionary and plot the pulse:# We can just plug in values for the parameters to get an actual pulse:
    parameters = {'ta': 200,
                  'tb': 2000,
                  'end': 4000,
                  'va': 2.2,
                  'vb': 3.0}
    # with these parameters, we can plot the pulse:
    plot(squarePulse, parameters)

Re-parametrizing is not difficult, either. We first choose a new set of parameters and then provide functions that can calculate the old/inner parameters from
the new/outer ones and collect them in a dictionary.
::

    mapping = {}
    mapping['ta'] = lambda ps: ps['start']
    mapping['tb'] = lambda ps: ps['start'] + ps['length']
    mapping['end'] = lambda ps: ps['pulse_length'] * 0.5
    mapping['value1'] = lambda ps: ps['value1']
    mapping['value2'] = lambda ps: ps['value2']

The mapping functions get called with one argument, a dictionary of *outer parameters*, and return a value for the *inner parameter*.
This example uses lambda functions, but you are free to use any python function.

We can now wrap our pulse in a `SequencePulseTemplate`:

::

    doubleSquare = SequencePulseTemplate([(squarePulse, mapping),
                                          (squarePulse, mapping)], # dictionaries with mapping functions from external parameters to subtemplate parameters
                                         ['start', 'length', 'value1', 'value2', 'pulse_length']) # declare the new template's external parameters

The first argument to the constructor is a list of tuples `(pulse template, mapping dictionary)`, the second argument is a list of the `SequencePulseTemplate`'s
*outer parameters*.

Just like with the simple pulse we can instantiate our new double pulse by providing parameters::

    params = dict(start=5,
                  length=20,
                  value1=10,
                  value2=15,
                  pulse_length=500)

    plot(doubleSquare, params)

Because `doubleSquare` is a pulse template we can just wrap it again, for convenience. The following block shows you how to define a nested pulse template
and how to use the low level plotting function to get the pulse shape as a numpy array::

    nested_mapping = dict(start=lambda ps: ps['start'],
                          length=lambda ps: ps['length'],
                          value1=lambda ps: 10, # You can fix parameters by giving a function that always returns the same value
                          value2=lambda ps: 20,
                          pulse_length=lambda ps: ps['pulse_length'] * 0.5)

    nested_pulse = SequencePulseTemplate([(doubleSquare, nested_mapping),
                                          (doubleSquare, nested_mapping)],
                                         ['start', 'length', 'pulse_length'])

    params2 = dict(start=10, length=100, pulse_length=1000)
    plot(nested_pulse, params2)

..    # Instead of calling the convenience plot function, we can also use the PlottingSequencer directly
    # This is also an instructive example on how to use sequencers.
    plotter = PlottingSequencer()
    plotter.push(nested_pulse, params2)
    times, voltages = plotter.render()
    plt.step(times, voltages)
    plt.show() # eh voila, a sequence of four pulses
