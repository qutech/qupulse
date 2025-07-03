Add metadata attribute to ``PulseTemplate`` and the keyword argument to  ``SequencePT``, ``RepetitionPT``, ``ForLoopPT`` and ``MappingPT``.
The metadata is intended for user data that does not influence the pulse itself. It is serialized with the pulse template but not part of the equality check because the field is mutable.

Currently, the only field that is used by qupulse itself is ``to_single_waveform``. When ``to_single_waveform='always'`` is set kin the metadata the corresponding pulse template is translated into a single waveform on program creation.
