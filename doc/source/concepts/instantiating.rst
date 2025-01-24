.. _instantiating:

Pulse Instantiation
-------------------

As already briefly mentioned in :ref:`pulsetemplates`, instantiation of pulses is the process of obtaining a hardware
interpretable representation of a concrete pulse ready for execution from the quite high-level :class:`.PulseTemplate`
object tree structure that defines parameterizable pulses in qupulse.

The entry point is the :meth:`.PulseTemplate.create_program` method of the :class:`.PulseTemplate` hierarchy.
It accepts the pulse parameters, and allows to rename and/or omit channels or measurements.
It checks that the provided parameters and mappings are consistent and meet the optionally defined parameter constraints of the pulse template.
The translation target is defined by the :class:`.ProgramBuilder` argument.

Each pulse template knows what program builder methods to call to translate itself.
For example, the :class:`.ConstantPulseTemplate` calls :meth:`.ProgramBuilder.hold_voltage` to hold a constant voltage for a defined amount of time while the :class:`.SequncePulseTemplate` forwards the program builder to the sub-templates in order.
The resulting program is completely backend dependent.

**Historically**, there was only a single program type :class:`.Loop` which is still the default output type.
As the time of this writing there is the additional :class:`.LinSpaceProgram` which allows for the efficient representation of linearly spaced voltage changes in arbitrary control structures. There is no established way to handle the latter yet.
The following describes handling of :class:`.Loop` object only via the :class:`.HardwareSetup`.

The :class:`.Loop` class was designed as a hardware-independent pulse program tree for waveform table based sequencers.
Therefore, the translation into a hardware specific format is a two-step process which consists of the loop object creation as a first step
and the transformation of that tree according to the needs of the hardware as a second step.
However, the AWGs became more flexibly programmable over the years as discussed in :ref:`awgs`.

The first step of this pulse instantiation is showcased in :ref:`/examples/02CreatePrograms.ipynb` where :meth:`.PulseTemplate.create_program` is used to create a :class:`.Loop` program.

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
However, this device supports increment commands which allow the efficient representation of linear voltage sweeps which is **not** possible with the :class:`.Loop` class.

The section :ref:`program` touches the ideas behind the current program implementations i.e. :class:`.Loop` and :class:`.LinSpaceProgram`.
