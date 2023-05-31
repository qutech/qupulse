.. _instantiating:

Pulse Instantiation
-------------------

As already briefly mentioned in :ref:`pulsetemplates`, instantiation of pulses is the process of obtaining a hardware
interpretable representation of a concrete pulse ready for execution from the quite high-level :class:`.PulseTemplate`
object tree structure that defines parameterizable pulses in qupulse.

This is a two-step process that involves

#. Inserting concrete parameter values and obtaining a hardware-independent pulse program tree
#. Flattening that tree, sampling and merging of leaf waveforms according to needs of hardware

This separation allows the first step to be performed in a hardware-agnostic way while the second step does not have
to deal with general functionality and can focus only on hardware-specific tasks. Step 1 is implemented in the
:meth:`.PulseTemplate.create_program` method of the :class:`.PulseTemplate` hierarchy. It checks parameter consistency
with parameter constraints and returns an object of type
:class:`.Loop` which represents a pulse as nested loops of atomic waveforms. This is another object tree structure
but all parameters (including repetition counts) have been substituted by the corresponding numeric values passed into
``create_program``. The :class:`.Loop` object acts as your reference to the instantiated pulse.
See :ref:`/examples/02CreatePrograms.ipynb` for an example on usage of :meth:`.PulseTemplate.create_program`.

The second step of the instantiation is performed by the hardware backend and transparent to the user. Upon registering
the pulse with the hardware backend via :meth:`qupulse.hardware.HardwareSetup.register_program`, the backend will determine which
hardware device is responsible for the channels defined in the pulse and delegate the :class:`.Loop` object to the
corresponding device driver. The driver will then sample the pulse waveforms with its configured sample rate, flatten
the program tree if required by the device and, finally, program the device and upload the sampled waveforms.

The flattening is device dependent because different devices allow for different levels of nested sequences and loops.

For example the Tabor Electronics WX2184C AWG supports two-fold nesting: waveforms into level-1 sequences, level-1 sequences
into level-2 sequences. In consequence, the program tree is flattened to depth two, i.e., for all tree paths of
larger depth, loops are unrolled and sequences of waveforms are merged into a single waveform until the target depth
is reached. Additionally, the AWG requires waveforms to have a minimal length. Any waveform that is shorter is merged
by the driver with its neighbors in the execution sequence until the minimum waveform length is reached. Further
optimizations and merges (or splits) of waveforms for performance are also possible.

In contrast, the Zurich Instruments HDAWG allows arbitrary nesting levels and is only limited by the instruction cache.

However, as already mentioned, the user does not have to be concerned about this in regular use of qupulse, since this
is dealt with transparently in the hardware backend.

The section :ref:`program` touches the ideas behind the current program implementation i.e. :class:`.Loop`.
