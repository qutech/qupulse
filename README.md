# qc-toolkit: Quantum Computing Toolkit
[![Coverage Status](https://coveralls.io/repos/qutech/qc-toolkit/badge.svg?branch=master&service=github)](https://coveralls.io/github/qutech/qc-toolkit?branch=master)
[![Build Status](https://travis-ci.org/qutech/qc-toolkit.svg?branch=master)](https://travis-ci.org/qutech/qc-toolkit)
[![Documentation Status](https://readthedocs.org/projects/qc-toolkit/badge/?version=latest)](http://qc-toolkit.readthedocs.org/en/latest/?badge=latest)

The qc-toolkit project aims to produce a software toolkit facilitating experiments involving pulse driven state manipulation of physical qubits.

It provides a high-level hardware-independent representation of pulses as well as means to translate this representation to hardware-specific device instructions and waveforms, execute these instructions and perform corresponding measurements.

Pulses can be assembled from previously defined subpulses, allowing easy construction of high-level from low-level pulses and re-use of previous work.
Additionally, all pulses are parameterizable allowing users to fine-tune and adapt pulse templates to specific hardware or functionality without redefining an entire pulse sequence. To ensure meaningful parameter values, constraints can be put on parameters on a per-pulse basis.  

## Status
Note that the project is still in somewhat early development and thus not feature-complete.

The qc-toolkit library is already used productively by the Bluhm research group at the 2nd Institute of Physics at the RWTH Aachen University. As such, some features - such as pulse definition - are mostly complete and tested and interfaces are expected to remain largely stable (or changes to be backward compatible).
However, it is still possible for existing portions of the code base to be redesigned if this will increase the usability long-term.
 
The current feature list is as follows:

- Definition of complex (arbitrarily deep nested and looped pulses) parameterized pulses in Python (including measurement windows)
- Mathematical expression evaluation (based on sympy) for parameter values and parameter constraints
- Serialization of pulses (to allow storing into permanent storage)
- Hardware model representation (prototype, work in progress)
- High-level pulse to hardware configuration and waveform translation routines 
- Hardware drivers for Tabor 200 AWG and AlazarTech DAC
- MATLAB interface to access qc-toolkit functionality

## Installation
qc-toolkit is developed using Python 3.5 but should also run on previous 3.3+ versions.

The package is installed with:
```
python3 setup.py install
```

The backend for TaborAWGs requires packages that can be found [here](https://git.rwth-aachen.de/qutech/python-TaborDriver).

The data acquisition backend for AlazarTech cards needs a package that unfortunately is not open source (yet). If you need it or have questions contact <simon.humpohl@rwth-aachen.de>.

The optional script *tests/utils/syntax_check.py* invokes pyflakes to perform a static code analysis, so pyflakes should be installed if its usage is intended.

## Documentation
You can find documentation on how to use this library on [readthedocs](http://qc-toolkit.readthedocs.io/en/latest/) and [IPython notebooks with examples in this repo](doc/source/examples)

## Folder Structure
The repository primarily consists of the folders `qctoolkit` and `tests`.

`qctoolkit` contains the entire source code of the project and is further partitioned into packages of related modules (i.e. a package folder `pulses` which contains all modules related to pulse representation and translation).

Contents of `tests` mirror the structure of `qctoolkit`. For every `<module>` somewhere in `qctoolkit` there should exist a `<module>Tests.py` in the corresponding subdirectory of `tests`.

