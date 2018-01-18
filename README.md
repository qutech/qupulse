# qc-toolkit: Quantum Computing Toolkit
[![Coverage Status](https://coveralls.io/repos/qutech/qc-toolkit/badge.svg?branch=master&service=github)](https://coveralls.io/github/qutech/qc-toolkit?branch=master)
[![Build Status](https://travis-ci.org/qutech/qc-toolkit.svg?branch=master)](https://travis-ci.org/qutech/qc-toolkit)
[![Documentation Status](https://readthedocs.org/projects/qc-toolkit/badge/?version=latest)](http://qc-toolkit.readthedocs.org/en/latest/?badge=latest)

The qc-toolkit project aims to produce a software toolkit facilitating experiments involving pulse driven state manipulation of physical qubits.
It will provide a hardware independent object representation of pulses and pulse templates as well as means to translate this representation to hardware instructions, execute these instructions and perform corresponding measurements.
Pulses may be as complex as specifying conditional branching/looping and the object representation features easy reuse of previously defined pulse templates.

## Status
Note that the project is in early development and thus neither feature-complete nor necessarily bug free. Additionally, interfaces and design decisions are subject to change.

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

