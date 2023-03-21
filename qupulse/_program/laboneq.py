from functools import singledispatch
from dataclasses import dataclass
from typing import Sequence, Mapping, Optional
from contextlib import contextmanager

from qupulse import ChannelID
from qupulse.pulses import ForLoopPT, RepetitionPT, SequencePT
from qupulse.pulses.pulse_template import AtomicPulseTemplate as AtomicPT
from qupulse.parameter_scope import Scope


ChannelMapping = Mapping[ChannelID, Optional[ChannelID]]


@dataclass
class TranslationContext:
    channels: Sequence[ChannelID]

    @contextmanager
    def with_for_loop(self, loop_index: str, loop_range: range):
        yield self
        raise NotImplementedError('magic')

    @contextmanager
    def with_repetition(self, repetition_count: int):
        yield self
        raise NotImplementedError('magic')


class LabOneQNode:
    def execute(self, experiment):
        pass


@dataclass
class Repetition:
    parameter: str

    def execute(self, experiment):
        with experiment.acquire_loop_rt():
            pass


@singledispatch
def to_laboneq(pt, context: TranslationContext, parameters: Scope):
    raise NotImplementedError(pt)


@to_laboneq.register
def _(pt: ForLoopPT, context: TranslationContext, parameters: Scope):
    loop_range = pt.loop_range.to_range(parameters)

    with context.with_for_loop(pt.loop_index, loop_range) as body_context:
        to_laboneq(pt.body, body_context, parameters)


@to_laboneq.register
def _(pt: SequencePT, context: TranslationContext, parameters: Scope):
    for template in pt.subtemplates:
        to_laboneq(template, context, parameters)


@to_laboneq.register
def _(pt: RepetitionPT, context: TranslationContext, parameters: Scope):
    repetition_count = pt.repetition_count.evaluate_in_scope(parameters)
    if repetition_count > 0:
        with context.with_repetition(repetition_count) as body_context:
            to_laboneq(pt.body, body_context, parameters)



@to_laboneq.register
def _(pt: AtomicPT, context: TranslationContext, parameters: Scope, channel_mapping: ChannelMapping):
    wf = pt.build_waveform(parameters, channel_mapping)
    if wf is not None:
        raise NotImplementedError("More magic")
    