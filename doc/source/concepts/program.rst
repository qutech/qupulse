.. _program:

Instantiated Pulse: Program
---------------------------

In qupulse an instantiated pulse template is called a program as it is something that an arbitrary waveform generator (AWG) can execute/playback.
It can be thought of as compact representation of a mapping :math:`\{t | 0 \le t \le t_{\texttt{duration}}\} \rightarrow \mathbb{R}^n` from the time while the program lasts :math:`t` to an n-dimensional voltage space :math:`\mathbb{R}^n`.
The dimensions are named by the channel names.

Programs are created by the :meth:`~.PulseTemplate.create_program` method of `PulseTemplate` which returns a hardware independent and un-parameterized representation.
The method takes a ``program_builder`` keyword argument that is propagated through the pulse template tree and thereby implements the visitor pattern.
If the argument is not passed :py:func:`~qupulse.program.default_program_builder()` is used instead which is :class:`.LoopBuilder` by default, i.e. the program created by default is of type :class:`.Loop`. The available program builders, programs and their constituents like :class:`.Waveform` and :class:`.VolatileRepetitionCount` are defined in th :mod:`qupulse.program` subpackage and it's submodules. There is a private ``qupulse._program`` subpackage that was used for more rapid iteration development and is slowly phased out. It still contains the hardware specific program representation for the tabor electronics AWG driver. Zurich instrument specific code has been factored into the separate package ``qupulse-hdawg``. Please refer to the reference and the docstrings for exact interfaces and implementation details.

The :class:`.Loop` default program is the root node of a tree of loop objects of arbitrary depth.
Each node consists of a repetition count and either a waveform or a sequence of nodes which are repeated that many times.
Iterations like the :class:`.ForLoopPT` cannot be represented natively but are unrolled into a sequence of items.
The repetition count is currently the only property of a program that can be defined as volatile. This means that the AWG driver tries to upload the program in a way, where the repetition count can quickly be changed. This is implemented via the ``VolatileRepetitionCount`` class.

A much more capable program format is :class:`.LinSpaceNode` which efficiently encodes linearly spaced sweeps in voltage space by utilizing increment commands. It is build via :class:`.LinSpaceBuilder`. Increment commands are available in the HDAWG command table.
