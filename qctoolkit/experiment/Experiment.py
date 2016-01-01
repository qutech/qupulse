from typing import Dict
import numpy as np
from itertools import product

from qctoolkit.pulses.measurements import Measurement
from qctoolkit.hardware.awgs import AWG
from qctoolkit.hardware.dacs import 
from qctoolkit.pulses.sequencing import Sequencer

"""Experiment class outline

An experiment needs access to the following things:

* Hardware: abstracted by proper driver classes
  * AWG
  * Acqusition device

* a pulse template, the parameters of which are swept
* a parameter space, which is scanned, optionally a scan order
* pulses that should be interspersed with measurements or executed before/after a "scanline"

internally it also needs
* a sequencer

The Experiment needs to figure out a number of things:

Preparation:
* render the pulse template using parameters from the parameter space
* make sure the pulses are uploaded to the awg
* figure out the lengths of pulses and arrange them in "scanlines"/acquisition lines
* add preparation pulses, triggering and such to the scanline
* configure the AWG so that the pulses are played in the right order
* figure out _when_ to measure and how to process the single measurements (downsampling, RPA, etc.)
  * create masks for the acquisition driver

Measurement:
* Trigger a measurement, handle the data from the DAQ device.
* re-arrange the measured/processed data to match the parameter space, this is the experiment result
"""

class Experiment():
    """An experiment implementing the behaviour that is oulined above."""
    def __init__(self, awg, dac, pulse_template: PulseTemplate, parameter_space: List[(str, np.ndarray)], downsampling=1):
        self.__template = pulse_template
        self.__parameter_space = parameter_space
        self.__sequencer = Sequencer()
        self.__awg = awg
        self.__dac = dac

    def prepare(self):
        # prepare parameter space, build tuples of all possible combinations in a defined order, last axis is fast.
        names, vectors = zip(*self.__parameter_space)
        vectors = tuple(vectors)
        # create array of tuples
        self.__parameter_tuples = np.array(list(it.product(*vectors)))
        # push everything on the sequencer
        # TODO: make several measurements instead of one
        # maybe things need to be pushed multiple times for downsampling
        # TODO: push some trigger element first
        for t in self.__parameter_tuples:
            parameters = {k: v for k,v in zip(names,t)}
            self.__sequencer.push(self.__template, parameters)
        self.__program = self.__sequencer.build()


        # now extract measurement windows somehow
        for block in self.__program:
            pass
            # TODO: the number of windows in a block should not change. they are put into an array and afterwards we need a routine that figures out what masks to use and how to configure them

