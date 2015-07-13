# qc-toolkit: Quantum Computing Toolkit
[![Build Status](https://travis-ci.org/qutech/qc-toolkit.svg)](https://travis-ci.org/qutech/qc-toolkit)


The qc-toolkit project aims to produce a software toolkit facilitating experiments (or productive use of) involving pulse driven state manipulation of physical qubits.
It will provide a hardware independent object representation of pulses and pulse templates as well as means to translate this representation to hardware instructions, execute these instructions and perform corresponding measurements.
Pulses may be as complex as specifying conditional branching/looping and the object representation features easy reuse of previously defined pulse templates.

## Status
Note that the project is in early development and thus neither feature-complete nor necessarily bug free. Additionally, interfaces and design decisions are subject to change.

## Installation
qc-toolkit is developed using Python 3.5 but should also run on previous 3.x versions (without guarantee) if the [typing module](https://github.com/JukkaL/typing) is installed.
Currently, there are no other required dependencies on external modules.
The optional script tests/utils/syntax_check.py invokes pyflakes to perform a static code analysis, so pyflakes should be installed if its usage is intended.

## Folder Structure
The repository primarily consists of the folders *src* and *tests*.

*src* contains the entire source code of the project and is further partitioned into packages of related modules (i.e. a package folder *pulses* which contains all modules related to pulse representation and translation).

Contents of *tests* mirror the structure of *src*. For every *<module>.py* somewhere in *src* there should exist a *<module>Tests.py* in the corresponding subdirectory of *tests*.
