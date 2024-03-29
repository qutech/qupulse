# qupulse: A Quantum compUting PULse parametrization and SEquencing framework - Documentation

This folder contains texts, configuration and scripts which are used to compile the documentation using [sphinx](http://www.sphinx-doc.org/en/stable/). It also contains usage examples for qupulse.
You may either build the documentation yourself or read it on [readthedocs](http://qc-toolkit.readthedocs.org/).[![Documentation Status](https://readthedocs.org/projects/qc-toolkit/badge/?version=latest)](http://qc-toolkit.readthedocs.org/en/latest/?badge=latest)


## Examples
In the subdirectory *examples* you can find various [Jupyter notebook](http://jupyter.org/) files providing some step-by-step examples of how qupulse can be used. These can be explored in an interactive fashion by running the *Jupyter notebook* application inside the folder. However, a static version will also be included in the documentation created with *sphinx*.

## Building the Documentation
To build the documentation, you will need [sphinx](http://www.sphinx-doc.org/en/stable/) and [nbsphinx](https://nbsphinx.readthedocs.org/) which, in turn, requires [pandoc](http://pandoc.org/).

The documentation is built by invoking `make <format>` inside the */doc* directory, where `<format>` is an output format supported by *sphinx*, e.g., `html`. The output will then be found in `/doc/build/<target>`.
