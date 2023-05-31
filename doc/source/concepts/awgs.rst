.. _hardware:

How qupulse models AWGs
-----------------------

This section is supposed to help you understand how qupulse sees AWGs and by extension help you understand the driver implementations in :py:mod:`~qupulse.hardware.awgs` and :py:mod:`~qupulse.hardware.feature_awg`.

When a program is uploaded to an arbitrary waveform generator (AWG) it needs to brought in a form that the hardware
understands.
Most AWGs consist of three significant parts:
 * The actual digital to analog converter (DAC) that outputs samples at a (semi-) fixed rate [1]_
 * A sequencer which tells the DAC what to do
 * Waveform memory which contains sampled waveforms in a format that the DAC understands

The sequencer feeds the data from the waveform memory to the DAC in the correct order.
Uploading a qupulse pulse to an AWG requires to sample the program, upload waveforms to the memory
and program the sequencer.

The interface exposed by the vendor to program the sequencer reaches from a simple table like for
Tektronix' AWG5000 series to some kind of complex domain specific language (DSL) like Zurich Instrument' sequencing C.

Basically all AWGs have some kind of limitations regarding the length of the waveform samples which is often of the
form :math:`n_{\texttt{samples}} = n_{\texttt{min}} + m \cdot n_{\texttt{div}}` with the minimal number of samples
:math:`n_{\texttt{min}}` and some divisor :math:`n_{\texttt{div}}`.

.. topic:: Implementation detail (might be outdated)

    Holding a voltage for a long time was often best accomplished by repeating a waveform of :math:`n_{\texttt{min}}` to save waveform memory.
    Earlier versions of qupulse required you to write your pulse in this way i.e. with a ``RepetitionPT``.
    Now qupulse contains the function ``qupulse._program._loop.roll_constant_waveform`` which detects long constant waveforms and rolls them into corresponding repetitions. This should be done by the hardware backend automatically.

.. [1] Some AWGs like the HDAWG can be programmed change the sample rate to a divisor of the "main" rate dynamically.
