# qupulse: A Quantum compUting PULse parametrization and SEquencing framework
[![Coverage Status](https://coveralls.io/repos/github/qutech/qupulse/badge.svg?branch=master)](https://coveralls.io/github/qutech/qupulse?branch=master)
[![Build Status](https://travis-ci.org/qutech/qupulse.svg?branch=master)](https://travis-ci.org/qutech/qupulse)
[![Documentation Status](https://readthedocs.org/projects/qupulse/badge/?version=latest)](http://qupulse.readthedocs.org/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/29915259.svg)](https://zenodo.org/badge/latestdoi/29915259)

The qupulse project aims to produce a software toolkit facilitating experiments involving pulse driven state manipulation of physical qubits.

It provides a high-level hardware-independent representation of pulses as well as means to translate this representation to hardware-specific device instructions and waveforms, execute these instructions and perform corresponding measurements.

Pulses can be assembled from previously defined subpulses, allowing easy construction of high-level from low-level pulses and re-use of previous work.
Additionally, all pulses are parameterizable allowing users to fine-tune and adapt pulse templates to specific hardware or functionality without redefining an entire pulse sequence. To ensure meaningful parameter values, constraints can be put on parameters on a per-pulse basis.  

## Status and stability
The qupulse library is used productively by the Quantum Technology Group at the 2nd Institute of Physics at the RWTH Aachen University. As such, some features - such as pulse definition - are mostly complete and tested and interfaces are expected to remain largely stable (or changes to be backward compatible). A key goal is that experiments should be repeatable with new versions of qupulse.
However, it is still possible for existing portions of the code base to be redesigned if this will increase the usability long-term.
 
The current feature list is as follows:

- Definition of complex (arbitrarily deep nested and looped pulses) parameterized pulses in Python (including measurement windows)
- Mathematical expression evaluation (based on sympy) for parameter values and parameter constraints
- Serialization of pulses (to allow storing into permanent storage)
- Hardware model representation
- High-level pulse to hardware configuration and waveform translation routines 
- Hardware drivers for Tabor Electronics, Tektronix and Zurich Instruments AWGs and AlazarTech Digitizers
 
Pending changes are tracked in the `changes.d` subdirectory and published in [`RELEASE_NOTES.rst`](RELEASE_NOTES.rst) on release using the tool `towncrier`.

### Removed features

The previous name of this package was qctoolkit. It was renamed in 2017 to highlight the pulse focus. The backward compatible alias was removed after the 0.9 release. Furthermore, this repository had a MATLAB interface for a longer time which was removed at the same time.

## Installation
qupulse is available on [PyPi](https://pypi.org/project/qupulse/) and the latest release can be installed by executing:
```sh
python -m pip install qupulse[default]
```
which will install all required and optional dependencies except for hardware support. qupulse version numbers follow the [Semantic Versioning](https://semver.org/) conventions.

Alternatively, the current development version of qupulse can be installed by executing
```sh
python -m pip install -e git+https://github.com/qutech/qupulse.git#egg=qupulse[default]
```
which will clone the github repository to `./src/qupulse` and do an editable/development install.

### Requirements and dependencies
qupulse requires at least Python 3.10 and is tested on 3.10, 3.11 and 3.12. It relies on some external Python packages as dependencies. 
We intentionally did not restrict versions of dependencies in the install scripts to not unnecessarily prevent usage of newer releases of dependencies that might be compatible. However, if qupulse does encounter problems with a particular dependency version please file an issue. 

The backend for TaborAWGs requires packages that can be found [here](https://git.rwth-aachen.de/qutech/python-TaborDriver).

The data acquisition backend for AlazarTech cards needs a package that unfortunately is not open source (yet). If you need it or have questions contact <simon.humpohl@rwth-aachen.de>.

## Documentation
You can find documentation on how to use this library on [readthedocs](https://qupulse.readthedocs.io/en/latest/) and [IPython notebooks with examples in this repo](doc/source/examples). You can build it locally with `hatch run docs:html` if you have pandoc installed.

### Folder Structure
The repository primarily consists of the folders `qupulse` (source code), `tests` and `doc`.

`qupulse` contains the entire Python source code of the project and is further partitioned the following packages of related packages 

- `pulses` which contains all modules related to pulse representation.
- `hardware` containing classes for hardware representation as well as hardware drivers
- `utils` containing miscellaneous utility modules or wrapping code for external libraries
- `program` contains general and hardware specific representations of instantiated (parameter free) pulses.
- `expression` contains the expression interface used by qupulse.

Contents of `tests` mirror the structure of `qupulse`. For every `<module>` somewhere in `qupulse` there should exist a `<module>Tests.py` in the corresponding subdirectory of `tests`.

## Development

`qupulse` uses `hatch` as development tool which provides a convenient interface for most development tasks. The following should work.

 - `hatch build`: Build wheel and source tarball
 - `hatch version X.X.X`: Set version
 - `hatch run docs:html`: Build documentation (requires pandoc)
 - `hatch run docs:clean-notebooks` to execute all example notebooks that start with 00-03 and clean all metadata. 
 - `hatch run changelog:draft` and `hatch run changelog:release` to preview or update the changelog.

## License

The current version of qupulse is available under the `LGPL-3.0-or-later` license. Versions up to and including 0.10 were licensed under the MIT license. If you require different licensing terms, please contact us to discuss your needs.
