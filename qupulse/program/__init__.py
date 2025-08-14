# SPDX-FileCopyrightText: 2014-2024 Quantum Technology Group and Chair of Software Engineering, RWTH Aachen University
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""This package contains the means to construct a program from a pulse template.

A program is an un-parameterized multichannel time to voltage relation. They are constructed by sequencing playback
commands which typically mean that an arbitrary waveform is played.

The arbitrary waveforms are defined in the :py:mod:`.waveforms` module.

:py:mod:`.transformation` contains useful transformations for waveforms which for example allow the
construction of virtual channels, i.e. linear combinations of channels from a set of other channes.

:py:mod:`.loop` contains the legacy program representation with is an aribtrariliy nested sequence/repetition structure
of waveform playbacks.

:py:mod:`.linspace` contains a more modern program representation to efficiently execute linearly spaced voltage sweeps
even if interleaved with constant segments.

:py:mod:`.volatile` contains the machinery to declare quickly changable program parameters. This functionality is stale
and was not used by the library authors for a long term. It is very useful for dynamic nuclear polarization which is not
used/required/possible with (purified) silicon samples.
"""

from qupulse.program.protocol import Program, ProgramBuilder
from qupulse.program.values import DynamicLinearValue, HardwareTime, \
    HardwareVoltage, RepetitionCount
from qupulse.program.waveforms import Waveform
from qupulse.program.transformation import Transformation
from qupulse.program.volatile import VolatileRepetitionCount


# backwards compatibility
# DEPRECATED but writing warning code for this is too complex
SimpleExpression = DynamicLinearValue


def default_program_builder() -> ProgramBuilder:
    """This function returns an instance of the default program builder class :class:`.LoopBuilder` in the default
    configuration.

    Returns:
        A program builder instance.
    """
    from qupulse.program.loop import LoopBuilder
    return LoopBuilder()
