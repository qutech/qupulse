.. _program:

Instantiated Pulse: Program
---------------------------

In qupulse an instantiated pulse template is called a program as it is something that an arbitrary waveform generator
(AWG) can execute/playback.
It is created by the ``create_program`` method of the pulse template which returns a hardware
independent representation which is of type ``Loop``.
This ``Loop`` object is the root node of a tree ``Loop``s of arbitrary depth.
Each node consists of a repetition count and either a waveform or a sequence of nodes which are repeated that many times.
Iterations like the ```ForLoopPT`` cannot be represented natively but are unrolled into a sequence of items.
The repetition count is currently the only property of a program that can be defined as volatile. This means that the AWG driver tries to upload the program in a way, where the repetition count can quickly be changed. This is implemented via the ```VolatileRepetitionCount`` class.

There is no description of the details of the program object here to avoid duplicated and outdated documentation.
The documentation is in the docstrings of the source code.
The program can be thought of as compact representation of a mapping :math:`\{t | 0 \le t \le t_{\texttt{duration}}} \rightarrow \mathbb{R}^n` from the time while the program lasts :math:´t´ to an n-dimensional voltage space :math:´\mathbb{R}^n´.
The dimensions are named by the channel names.

The ``Loop`` class and its constituents ``Waveform`` and ``VolatileRepetitionCount`` are defined in the ``qupulse.program`` subpackage and it's submodules.
The private subpackage ``qupulse._program`` contains AWG driver internals that can change with any release, for example a
transpiler to Zurich Instruments sequencing C in ``qupulse._program.seqc``.
