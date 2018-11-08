# qupulse: A Quantum compUting PULse parametrization and SEquencing framework
[![Coverage Status](https://coveralls.io/repos/github/qutech/qupulse/badge.svg?branch=master)](https://coveralls.io/github/qutech/qupulse?branch=master)
[![Build Status](https://travis-ci.org/qutech/qupulse.svg?branch=master)](https://travis-ci.org/qutech/qupulse)
[![Documentation Status](https://readthedocs.org/projects/qupulse/badge/?version=latest)](http://qupulse.readthedocs.org/en/latest/?badge=latest)

The qupulse project aims to produce a software toolkit facilitating experiments involving pulse driven state manipulation of physical qubits.

It provides a high-level hardware-independent representation of pulses as well as means to translate this representation to hardware-specific device instructions and waveforms, execute these instructions and perform corresponding measurements.

Pulses can be assembled from previously defined subpulses, allowing easy construction of high-level from low-level pulses and re-use of previous work.
Additionally, all pulses are parameterizable allowing users to fine-tune and adapt pulse templates to specific hardware or functionality without redefining an entire pulse sequence. To ensure meaningful parameter values, constraints can be put on parameters on a per-pulse basis.  

## Status
Note that the project is still in development and thus not feature-complete.

The qupulse library is already used productively by the Quantum Technology Group at the 2nd Institute of Physics at the RWTH Aachen University. As such, some features - such as pulse definition - are mostly complete and tested and interfaces are expected to remain largely stable (or changes to be backward compatible).
However, it is still possible for existing portions of the code base to be redesigned if this will increase the usability long-term.
 
The current feature list is as follows:

- Definition of complex (arbitrarily deep nested and looped pulses) parameterized pulses in Python (including measurement windows)
- Mathematical expression evaluation (based on sympy) for parameter values and parameter constraints
- Serialization of pulses (to allow storing into permanent storage)
- Hardware model representation (prototype, work in progress)
- High-level pulse to hardware configuration and waveform translation routines 
- Hardware drivers for Tabor Electronics AWGs and AlazarTech Digitizers
- MATLAB interface to access qupulse functionality

## Installation
qupulse is available on [PyPi](https://pypi.org/project/qupulse/) and the latest release can be installed by executing:
```
pip3 install qupulse
```
qupulse version numbers follow the [Semantic Versioning](https://semver.org/) conventions.

Alternatively, the current development version of qupulse can be installed by executing in the cloned repository root folder: 
```
pip3 install .
```

qupulse is developed using Python 3.6 and tested on 3.5 - 3.7 It relies on some external Python packages as dependencies; 
`requirements.txt` lists the versions of these qupulse is developed against. 
We intentionally did not restrict versions of dependencies in the install scripts to not unnecessarily prevent usage of
newer releases of dependencies that might be compatible. However, if qupulse does encounter problems with a particular dependency version,
try installing the version listed in `requirements.txt`.   

The backend for TaborAWGs requires packages that can be found [here](https://git.rwth-aachen.de/qutech/python-TaborDriver).

The data acquisition backend for AlazarTech cards needs a package that unfortunately is not open source (yet). If you need it or have questions contact <simon.humpohl@rwth-aachen.de>.

The optional script *tests/utils/syntax_check.py* invokes pyflakes to perform a static code analysis, so pyflakes should be installed if its usage is intended.

## Documentation
You can find documentation on how to use this library on [readthedocs](http://qc-toolkit.readthedocs.io/en/latest/) and [IPython notebooks with examples in this repo](doc/source/examples)

## Folder Structure
The repository primarily consists of the folders `qupulse` (toolkit core code) and `tests` (toolkit core tests). Additional parts of the project reside in `MATLAB` (MATLAB interface) and `doc` (configuration and source files to build documentation)  

`qupulse` contains the entire Python source code of the project and is further partitioned the following packages of related modules 

- `pulses` which contains all modules related to pulse representation.
- `hardware` containing classes for hardware representation as well as hardware drivers
- `utils` containing miscellaneous utility modules or wrapping code for external libraries


Contents of `tests` mirror the structure of `qupulse`. For every `<module>` somewhere in `qupulse` there should exist a `<module>Tests.py` in the corresponding subdirectory of `tests`.

