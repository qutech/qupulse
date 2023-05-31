.. _program:

Instantiated Pulse: Program
---------------------------

In qupulse an instantiated pulse template is called a program as it is something that an arbitrary waveform generator
(AWG) can execute/playback. It is created by the `create_program` method of the pulse template which returns a hardware
independent representation which is currently of type ``Loop``. Opposed to the `PulseTemplate` interfaces the interface of the program is currently not covered by the qupulse backward compatibility and stability guarantee.
This is reflected by the fact that it lives in the private module ``qupulse._program._loop``.

There is no description of the details of the program object here to avoid outdated documentation.
The documentation is in the docstrings of the source code.
The program can be thought of as compact representation of a mapping :math:`\{t | 0 \le t \le t_{\texttt{duration}}} \rightarrow \mathbb{R}^n` from the time while the program lasts :math:´t´ to an n-dimensional voltage space :math:´\mathbb{R}^n´.
The dimensions are named by the channel names.

The subpackage ``_qupulse._program`` also contains hardware specific translations of the programs for example a
transpiler to Zurich Instruments sequencing C in ``_qupulse._program.seqc``.
