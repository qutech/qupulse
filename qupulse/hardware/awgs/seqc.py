from typing import Optional, cast, Union, Sequence, Dict, MutableSequence, Iterator, Generator, Tuple, Callable
import abc
import itertools

import numpy as np

from qupulse._program.waveforms import Waveform
from qupulse._program._loop import Loop


class BinaryWaveform:
    __slots__ = ('data', 'channel_mask', 'marker_mask')

    def __init__(self, data: np.ndarray,
                 channel_mask: Tuple[bool, bool],
                 marker_mask: Tuple[bool, bool]):
        """ always use both channels?

        Args:
            data: data as returned from zhinst.utils.convert_awg_waveform
            channel_mask: which channels are given
            marker_mask:
        """
        self.data = data
        self.channel_mask = channel_mask
        self.marker_mask = marker_mask

        # needed to be hashable
        self.data.flags.writable = False

    def __len__(self):
        return len(self.data) // (sum(self.channel_mask) + any(self.marker_mask))

    def __eq__(self, other):
        return (self.channel_mask == other.channel_mask and
                self.marker_mask == other.marker_mask and
                np.array_equal(self.data, other.data))

    def __hash__(self):
        return hash((self.channel_mask, self.marker_mask, bytes(self.data)))


def find_sharable_waveforms(node_cluster: Sequence['SEQCNode']) -> Optional[Sequence[bool]]:
    """Expects nodes to have a compatible stepping

    TODO: encode in type system?
    """
    waveform_playbacks = list(node_cluster[0].iter_waveform_playbacks())

    candidates = [True] * len(waveform_playbacks)

    for node in itertools.islice(node_cluster, 1, None):
        candidates_left = False
        for idx, (wf, node_wf) in enumerate(zip(waveform_playbacks, node.iter_waveform_playbacks())):
            if candidates[idx]:
                candidates[idx] = wf == node_wf
            candidates_left = candidates_left or candidates[idx]

        if not candidates_left:
            return None

    return candidates


def mark_sharable_waveforms(node_cluster: Sequence['SEQCNode'], sharable_waveforms: Sequence[bool]):
    for node in node_cluster:
        for sharable, wf_playback in zip(sharable_waveforms, node.iter_waveform_playbacks()):
            wf_playback.shared = True


def to_node_clusters(loop: Loop, loop_to_seqc_kwargs: dict) -> Sequence[Sequence['SEQCNode']]:
    """transform to seqc recursively noes and cluster them if they have compatible stepping"""
    assert len(loop) > 1

    node_clusters = []
    last_node = loop_to_seqc(loop[0], **loop_to_seqc_kwargs)
    current_nodes = [last_node]

    # compress all nodes in clusters of the same stepping
    for child in itertools.islice(loop, 1, None):
        current_node = loop_to_seqc(child, **loop_to_seqc_kwargs)

        if last_node.same_stepping(current_node):
            current_nodes.append(current_node)
        else:
            node_clusters.append(current_nodes)
            current_nodes = [current_node]

        last_node = current_node
    node_clusters.append(current_nodes)
    return node_clusters


def loop_to_seqc(loop: Loop,
                 min_repetitions_for_for_loop: int,
                 min_repetitions_for_shared_wf: int,
                 waveform_to_bin: Callable[[Waveform], BinaryWaveform]) -> 'SEQCNode':
    assert min_repetitions_for_for_loop >= min_repetitions_for_shared_wf
    # At which point do we switch from indexed to shared

    if loop.is_leaf():
        node = WaveformPlayback(waveform_to_bin(loop.waveform))

    elif len(loop) == 1:
        node = loop_to_seqc(loop[0],
                            min_repetitions_for_for_loop=min_repetitions_for_for_loop,
                            min_repetitions_for_shared_wf=min_repetitions_for_shared_wf,
                            waveform_to_bin=waveform_to_bin)

    else:
        node_clusters = to_node_clusters(loop, dict(min_repetitions_for_for_loop=min_repetitions_for_for_loop,
                                                    min_repetitions_for_shared_wf=min_repetitions_for_shared_wf,
                                                    waveform_to_bin=waveform_to_bin))

        seqc_nodes = []

        # identify shared waveforms in node clusters
        for node_cluster in node_clusters:
            if len(node_cluster) < min_repetitions_for_for_loop:
                seqc_nodes.extend(node_cluster)

            else:
                if len(node_cluster) >= min_repetitions_for_shared_wf:
                    sharable_waveforms = find_sharable_waveforms(node_cluster)
                    if sharable_waveforms:
                        mark_sharable_waveforms(node_cluster, sharable_waveforms)

                seqc_nodes.append(SteppingRepeat(node_cluster))

        node = Scope(seqc_nodes)

    if loop.repetition_count != 1:
        return Repeat(scope=node, repetition_count=loop.repetition_count)
    else:
        return node


class SEQCNode(metaclass=abc.ABCMeta):
    INDENTATION = '  '

    @abc.abstractmethod
    def samples(self) -> int:
        pass

    @abc.abstractmethod
    def same_stepping(self, other: 'SEQCNode'):
        pass

    @abc.abstractmethod
    def iter_waveform_playbacks(self) -> Iterator['WaveformPlayback']:
        pass

    @abc.abstractmethod
    def _visit_nodes(self, waveform_manager):
        """push all concatenated waveforms in the waveform manager"""

    @abc.abstractmethod
    def to_source_code(self, waveform_manager, node_name_generator: Iterator[str], line_prefix: str, pos_var_name: str):
        """besides creating the source code, this function registers all needed waveforms to the program manager
        1. shared waveforms
        2. concatenated waveforms in the correct order

        Args:
            waveform_manager:
            node_name_generator: generates unique names of nodes
            line_prefix:
            pos_var_name:

        Returns:

        """


class Scope(SEQCNode):
    """Sequence of nodes"""

    __slots__ = ('nodes',)

    def __init__(self, nodes: Sequence[SEQCNode] = ()):
        self.nodes = list(nodes)

    def samples(self):
        return sum(node.samples() for node in self.nodes)

    def iter_waveform_playbacks(self) -> Iterator['WaveformPlayback']:
        for node in self.nodes:
            yield from node.iter_waveform_playbacks()

    def same_stepping(self, other: 'Scope'):
        return type(other) is Scope and all(n1.same_stepping(n2) for n1, n2 in zip(self.nodes, other.nodes))

    def _visit_nodes(self, waveform_manager):
        for node in self.nodes:
            node._visit_nodes(waveform_manager)

    def to_source_code(self, waveform_manager, node_name_generator: Iterator[str], line_prefix: str, pos_var_name: str):
        for node in self.nodes:
            yield from node.to_source_code(waveform_manager,
                                           line_prefix=line_prefix + self.INDENTATION,
                                           pos_var_name=pos_var_name,
                                           node_name_generator=node_name_generator)


class Repeat(SEQCNode):
    """
    stepping: if False resets the pos to initial value after each iteration"""

    def __init__(self, repetition_count: int, scope: SEQCNode):
        assert repetition_count > 1
        self.repetition_count = repetition_count
        self.scope = scope

    def samples(self):
        return self.scope.samples()

    def same_stepping(self, other: 'Repeat'):
        return (type(self) == type(other) and
                self.repetition_count == other.repetition_count and
                self.scope.same_stepping(other.scope))

    def iter_waveform_playbacks(self) -> Iterator['WaveformPlayback']:
        return self.scope.iter_waveform_playbacks()

    def _visit_nodes(self, waveform_manager):
        self.scope._visit_nodes(waveform_manager)

    def to_source_code(self, waveform_manager, node_name_generator: Iterator[str], line_prefix: str, pos_var_name: str):
        """

        Args:
            waveform_manager:
            line_prefix:
            node_name_generator: Used to name the initial position variable
            pos_var_name:

        Returns:

        """
        body_prefix = line_prefix + self.INDENTATION
        node_name = next(node_name_generator)
        
        initial_position_name = 'init_pos_%s' % node_name
        
        # store initial position
        yield '{line_prefix}var {init_pos_name} = {pos_var_name};'.format(line_prefix=line_prefix,
                                                                          init_pos_name=initial_position_name,
                                                                          pos_var_name=pos_var_name)

        yield '{line_prefix}repeat({repetition_count}) {{ // {node_name}'.format(line_prefix=line_prefix,
                                                                                 repetition_count=self.repetition_count,
                                                                                 node_name=node_name)
        yield ('{body_prefix}{pos_var_name} = {init_pos_name};'
               ' // set back on each iteration').format(body_prefix=body_prefix,
                                                        pos_var_name=pos_var_name,
                                                        init_pos_name=initial_position_name)
        yield from self.scope.to_source_code(waveform_manager,
                                             line_prefix=body_prefix, pos_var_name=pos_var_name,
                                             node_name_generator=node_name_generator)
        yield '{line_prefix}}} // repetition {node_name} end'.format(line_prefix=line_prefix, node_name=node_name)


class SteppingRepeat(SEQCNode):
    def __init__(self, node_cluster: Sequence[SEQCNode]):
        self.node_cluster = node_cluster

    def samples(self) -> int:
        return self.repetition_count * self.node_cluster[0].samples()

    @property
    def repetition_count(self):
        return len(self.node_cluster)

    def iter_waveform_playbacks(self) -> Iterator['WaveformPlayback']:
        for node in self.node_cluster:
            yield from node.iter_waveform_playbacks()

    def same_stepping(self, other: 'SteppingRepeat'):
        return (type(other) is SteppingRepeat and
                self.repetition_count == other.repetition_count and
                self.node_cluster[0].same_stepping(other.node_cluster[0]))

    def _visit_nodes(self, waveform_manager):
        for node in self.node_cluster:
            node._visit_nodes(waveform_manager)

    def to_source_code(self, waveform_manager, node_name_generator: Iterator[str], line_prefix: str, pos_var_name: str):
        body_prefix = line_prefix + self.INDENTATION
        node_name = next(node_name_generator)

        yield ('{line_prefix}repeat({repetition_count}) {{ '
               '// stepping repeat {node_name}').format(line_prefix=line_prefix,
                                                        repetition_count=self.repetition_count)
        yield from self.node_cluster[0].to_source_code(waveform_manager,
                                                       line_prefix=body_prefix, pos_var_name=pos_var_name,
                                                       node_name_generator=node_name_generator)

        # register remaining concatenated waveforms
        for node in itertools.islice(self.node_cluster, 1, None):
            node._visit_nodes(waveform_manager)

        yield '{line_prefix}}} // end stepping repeat {node_name}'.format(line_prefix=line_prefix,
                                                                          node_name=node_name)


class WaveformPlayback(SEQCNode):
    __slots__ = ('waveform', 'shared')

    def __init__(self, waveform: BinaryWaveform, shared: bool = False):
        self.waveform = waveform
        self.shared = shared

    def samples(self):
        if self.shared:
            return 0
        else:
            return len(self.waveform)

    def same_stepping(self, other: 'WaveformPlayback'):
        same_type = type(self) is type(other) and self.shared == other.shared
        if self.shared:
            return same_type and self.waveform == other.waveform
        else:
            return same_type and len(self.waveform) == len(other.waveform)

    def iter_waveform_playbacks(self) -> Iterator['WaveformPlayback']:
        yield self

    def _visit_nodes(self, waveform_manager):
        if not self.shared:
            waveform_manager.request_concatenated(self.waveform)

    def to_source_code(self, waveform_manager,
                       line_prefix: str, pos_var_name: str,
                       initial_pos_var_value: Optional[int]):
        if self.shared:
            yield 'playWaveform({waveform});'.format(waveform=waveform_manager.request_shared(self.waveform))
        else:
            wf_name = waveform_manager.request_concatenated(self.waveform)
            wf_len = len(self.waveform)
            yield 'playWaveformIndexed({wf_name}, pos, {wf_len}); pos = pos + {wf_len};'.format(wf_name=wf_name,
                                                                                                wf_len=wf_len)
