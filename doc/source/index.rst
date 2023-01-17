.. qupulse documentation master file, created by
   sphinx-quickstart on Mon Aug 10 09:57:22 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to qupulse's documentation!
======================================

``qupulse`` is a python package to write, manage and playback arbitrarily nested quantum control pulses. This documentation contains concept explanations, jupyter notebook examples and the automatically generated API reference. The API reference does not cover parts of qupulse that are explicitly considered an implementation detail like ``qupulse._program``.

You are encouraged to read the concept explanations and interactively explore the linked examples. To do this you can install qupulse via ``python -m pip install -e git+https://github.com/qutech/qupulse.git#egg=qupulse[default]`` which will clone the qupulse into ``./src/qupulse``. You can find the examples in ``doc/source/examples`` and open them with jupyter, Spyder or another IDE of your choice.

There is a :ref:`learners guide <learners_guide>` available to help with an efficient exploration of qupulse's features.

Contents:

.. toctree::
   :maxdepth: 4
   :numbered:

   concepts/concepts
   examples/examples
   _autosummary/qupulse
   learners_guide

qupulse API Documentation
=========================

.. autosummary::
   :recursive:
   :toctree: _autosummary
   :template: autosummary/package.rst

    qupulse

.. qupulse API Documentation <qupulse>

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

