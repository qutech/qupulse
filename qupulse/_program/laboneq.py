from functools import singledispatch
from dataclasses import dataclass
from typing import Sequence, Mapping, Optional, List, Tuple
from contextlib import contextmanager

from qupulse import ChannelID
from qupulse.pulses import ForLoopPT, RepetitionPT, SequencePT, MappingPT, ConstantPT
from qupulse.pulses.pulse_template import AtomicPulseTemplate as AtomicPT
from qupulse.parameter_scope import Scope


ChannelMapping = Mapping[ChannelID, Optional[ChannelID]]


@dataclass
class LabOneQProgram:
    parameters =


@dataclass
class TranslationContext:
    channels: Sequence[ChannelID]
    parameters: Scope

    active_sweeps: List['Sweep']

    body: List['LabOneQNode']

    @contextmanager
    def with_for_loop(self, loop_index: str, loop_range: range):
        self.active_sweeps.append(Sweep(loop_index, loop_range, [], []))
        inner = TranslationContext(
            channels=self.channels,
            parameters=self.parameters,
            active_sweeps=self.active_sweeps,
            body=[]
        )
        yield inner
        sweep = self.active_sweeps.pop()
        if sweep.parameters:
            self.body.append(sweep)
        else:
            raise NotImplementedError('Put repetition here since sweep parameter was not used')

    @contextmanager
    def with_repetition(self, repetition_count: int):
        inner = TranslationContext(
            channels=self.channels,
            active_sweeps=self.active_sweeps,
            body=[]
        )
        yield inner
        if inner.body:
            self.body.append(Repetition(count=repetition_count, body=inner.body))


class LabOneQNode:
    def execute(self, experiment):
        raise NotImplementedError()


@dataclass
class Sweep(LabOneQNode):
    loop_index: str
    loop_range: range
    parameters: List

    body: List[LabOneQNode]


@dataclass
class Repetition(LabOneQNode):
    count: int

    body: List[LabOneQNode]

    def execute(self, experiment):
        with experiment.acquire_loop_rt():
            raise NotImplementedError()


@singledispatch
def to_laboneq(pt, context: TranslationContext, parameters: Scope):
    raise NotImplementedError(pt)


@to_laboneq.register
def _(pt: ForLoopPT, context: TranslationContext):
    loop_range = pt.loop_range.to_range(context.parameters)

    with context.with_for_loop(pt.loop_index, loop_range) as body_context:
        to_laboneq(pt.body, body_context, context.parameters)


@to_laboneq.register
def _(pt: SequencePT, context: TranslationContext):
    for template in pt.subtemplates:
        to_laboneq(template, context, context.parameters)


@to_laboneq.register
def _(pt: RepetitionPT, context: TranslationContext):
    

    repetition_count = pt.get_repetition_count_value(context.parameters)
    if repetition_count > 0:
        with context.with_repetition(repetition_count) as body_context:
            to_laboneq(pt.body, body_context, parameters)


@to_laboneq.register
def _(pt: MappingPT, context: TranslationContext):
    raise NotImplementedError()


@to_laboneq.register
def _(pt: ConstantPT, context: TranslationContext, parameters: Scope, channel_mapping: ChannelMapping):

    wf = pt.build_waveform(parameters, channel_mapping)
    if wf is not None:
        raise NotImplementedError("More magic")
