from typing import Optional, cast, Union, Sequence, Dict, Iterator, Tuple, Callable, NamedTuple, MutableMapping, Mapping
from types import MappingProxyType
import abc
import itertools
import inspect
import glob
import os.path
from collections import OrderedDict

import numpy as np

from qupulse._program.waveforms import Waveform
from qupulse._program._loop import Loop

try:
    import zhinst.utils
except ImportError:
    zhinst = None


class BinaryWaveform:
    """This class represents a sampled waveform in the native HDAWG format as returned
    by zhinst.utils.convert_awg_waveform.

    BinaryWaveform.data can be uploaded directly to {device]/awgs/{awg}/waveform/waves/{wf}

    `to_csv_compatible_table` can be used to create a compatible compact csv file (with marker data included)
    """
    __slots__ = ('data',)

    def __init__(self, data: np.ndarray):
        """ always use both channels?

        Args:
            data: data as returned from zhinst.utils.convert_awg_waveform
        """
        n_quantum, remainder = divmod(data.size, 3 * 16)
        assert n_quantum > 1, "Waveform too short"
        assert remainder == 0, "Waveform has not a valid length"
        assert data.dtype is np.uint16
        assert np.all(data[2::3] < 16), "invalid marker data"

        self.data = data
        self.data.flags.writeable = False

    @property
    def ch1(self):
        return self.data[::3]

    @property
    def ch2(self):
        return self.data[1::3]

    @property
    def marker_data(self):
        return self.data[2::3]

    @property
    def markers_ch1(self):
        return np.bitwise_and(self.marker_data, 0b0011)

    @property
    def markers_ch2(self):
        return np.bitwise_and(self.marker_data, 0b1100)

    @classmethod
    def from_sampled(cls, ch1: Optional[np.ndarray], ch2: Optional[np.ndarray],
                     markers: Tuple[Optional[np.ndarray], Optional[np.ndarray],
                                    Optional[np.ndarray], Optional[np.ndarray]]) -> 'BinaryWaveform':
        """Combines the sampled and scaled waveform data into a single binary compatible waveform

        Args:
            ch1: sampled waveform scaled to full range (-1., 1.)
            ch2: sampled waveform scaled to full range (-1., 1.)
            markers: (ch1_front_marker, ch1_dio_marker, ch2_front_marker, ch2_dio_marker)

        Returns:

        """
        all_input = (ch1, ch2, *markers)
        assert any(x is not None for x in all_input)
        size = next(x.size for x in all_input if x is not None)
        if ch1 is None:
            ch1 = np.zeros(size)
        if ch2 is None:
            ch2 = np.zeros(size)
        marker_data = np.zeros(size, dtype=np.uint16)
        for idx, marker in enumerate(markers):
            if marker is not None:
                marker_data += (marker > 0) * 2**idx
        return cls(zhinst.utils.convert_awg_waveform(ch1, ch2, marker_data))

    def __len__(self):
        return self.data.size // 3

    def __eq__(self, other):
        return np.array_equal(self.data, other.data)

    def __hash__(self):
        return hash(bytes(self.data))

    def to_csv_compatible_table(self):
        """The integer values in that file should be 18-bit unsigned integers with the two least significant bits
        being the markers. The values are mapped to 0 => -FS, 262143 => +FS, with FS equal to the full scale.

        >>> np.savetxt(waveform_dir, binary_waveform.to_csv_compatible_table(), fmt='%u')
        """
        table = np.zeros((len(self), 2), dtype=np.uint32)
        table[:, 0] = self.ch1
        table[:, 1] = self.ch2
        np.left_shift(table, 2, out=table)
        table[:, 0] += self.markers_ch1
        table[:, 1] += self.markers_ch2

        return table


class ConcatenatedWaveform:
    def __init__(self):
        self._concatenated = []
        self._as_binary = None

    def __bool__(self):
        return bool(self._concatenated)

    def is_finalized(self):
        return self._as_binary is not None or self._concatenated is None

    def as_binary(self) -> Optional[BinaryWaveform]:
        assert self.is_finalized()
        return self._as_binary

    def append(self, binary_waveform):
        assert not self.is_finalized()
        self._concatenated.append(binary_waveform)

    def finalize(self):
        assert not self.is_finalized()
        if self._concatenated:
            concatenated_data = np.concatenate([wf.data for wf in self._concatenated], order='F')
            self._as_binary = BinaryWaveform(concatenated_data)
        else:
            self._concatenated = None

    def clear(self):
        if self._concatenated is None:
            self._concatenated = []
        else:
            self._concatenated.clear()
        self._as_binary = None


class WaveformMemory:
    """Global waveform "memory" representation (currently the file system)"""
    CONCATENATED_WAVEFORM_TEMPLATE = '{program_name}_concatenated_waveform'
    SHARED_WAVEFORM_TEMPLATE = '{program_name}_shared_waveform_{hash}'
    WF_PLACEHOLDER_TEMPLATE = '*{id}*'
    FILE_NAME_TEMPLATE = '{hash}.csv'

    _WaveInfo = NamedTuple('_WaveInfo', [('wave_name', str),
                                         ('wave_placeholder', str),
                                         ('file_name', str),
                                         ('binary_waveform', BinaryWaveform)])

    def __init__(self):
        self.shared_waveforms = OrderedDict()  # type: MutableMapping[BinaryWaveform, set]
        self.concatenated_waveforms = OrderedDict()  # type: MutableMapping[str, ConcatenatedWaveform]

    def _shared_waveforms_iter(self) -> Iterator[_WaveInfo]:
        for wf, program_set in self.shared_waveforms.items():
            if program_set:
                wave_hash = hash(wf)
                wave_name = self.SHARED_WAVEFORM_TEMPLATE.format(program_name='_'.join(program_set),
                                                                 hash=wave_hash)
                wave_placeholder = self.WF_PLACEHOLDER_TEMPLATE.format(id=id(program_set))
                file_name = self.FILE_NAME_TEMPLATE.format(hash=wave_hash)
                yield self._WaveInfo(wave_name, wave_placeholder, file_name, wf)

    def _concatenated_waveforms_iter(self) -> Iterator[_WaveInfo]:
        for program_name, concatenated_waveform in self.concatenated_waveforms.items():
            if concatenated_waveform:
                wave_hash = hash(concatenated_waveform.as_binary())
                wave_placeholder = self.WF_PLACEHOLDER_TEMPLATE.format(id=id(concatenated_waveform))
                wave_name = self.CONCATENATED_WAVEFORM_TEMPLATE.format(program_name=program_name)
                file_name = self.FILE_NAME_TEMPLATE.format(hash=wave_hash)
                yield self._WaveInfo(wave_name, wave_placeholder, file_name, concatenated_waveform.as_binary())

    def waveform_name_translation(self) -> Dict[str, str]:
        """replace place holders of complete seqc program with

        >>> waveform_name_translation = waveform_memory.waveform_name_translation()
        >>> seqc_program = seqc_program.translate(waveform_name_translation)
        """
        translation = {}
        for wave_info in self._shared_waveforms_iter():
            translation[wave_info.wave_placeholder] = wave_info.wave_name

        for wave_info in self._concatenated_waveforms_iter():
            translation[wave_info.wave_placeholder] = wave_info.wave_name
        return translation

    def waveform_declaration(self) -> str:
        """Produces a string that declares all needed waveforms.
        It is needed to know the waveform index in case we want to update a waveform during playback."""
        declarations = []
        for wave_info in self._concatenated_waveforms_iter():
            declarations.append(
                'wave {wave_name} = "{file_name}";'.format(wave_name=wave_info.wave_name,
                                                           file_name=wave_info.file_name.replace('.csv', ''))
            )

        for wave_info in self._shared_waveforms_iter():
            declarations.append(
                'wave {wave_name} = "{file_name}";'.format(wave_name=wave_info.wave_name,
                                                           file_name=wave_info.file_name.replace('.csv', ''))
            )
        return '\n'.join(declarations)

    def sync_to_file_system(self, path, delete=True, write_all=False):
        to_save = {wave_info.file_name: wave_info.binary_waveform
                   for wave_info in itertools.chain(self._concatenated_waveforms_iter(),
                                                    self._shared_waveforms_iter())}

        for file_name in glob.glob(os.path.join(path, '*.csv')):
            if file_name in to_save:
                if not write_all:
                    to_save.pop(file_name)
            elif delete:
                try:
                    os.remove(file_name)
                except OSError:
                    # TODO: log
                    pass

        for file_name, binary_waveform in to_save.items():
            table = binary_waveform.to_csv_compatible_table()
            np.savetxt(file_name, table, '%u')


class ProgramWaveformManager:
    """Manages waveforms of a program"""
    def __init__(self, name, memory: WaveformMemory):
        self._program_name = name
        self._memory = memory

        assert self._program_name not in self._memory.concatenated_waveforms
        assert all(self._program_name not in programs for programs in self._memory.shared_waveforms.values())
        self._memory.concatenated_waveforms[self._program_name] = ConcatenatedWaveform()

    def clear_requested(self):
        for programs in self._memory.shared_waveforms.values():
            programs.discard(self._program_name)
        self._memory.concatenated_waveforms[self._program_name].clear()

    def request_shared(self, binary_waveform: BinaryWaveform) -> str:
        """Register waveform if not already registered and return a unique identifier placeholder.

        The unique identifier currently is computed from the id of the set which stores all programs using this
        waveform.
        """
        program_set = self._memory.shared_waveforms.setdefault(binary_waveform, set())
        program_set.add(self._program_name)
        return self._memory.WF_PLACEHOLDER_TEMPLATE.format(id=id(program_set))

    def request_concatenated(self, binary_waveform: BinaryWaveform) -> str:
        """Append the waveform to the concatenated waveform"""
        bin_wf_list = self._memory.concatenated_waveforms[self._program_name]
        bin_wf_list.append(binary_waveform)
        return self._memory.WF_PLACEHOLDER_TEMPLATE.format(id=id(bin_wf_list))

    def finalize(self):
        self._memory.concatenated_waveforms[self._program_name].finalize()

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
            if sharable:
                wf_playback.shared = True


def to_node_clusters(loop: Union[Sequence[Loop], Loop], loop_to_seqc_kwargs: dict) -> Sequence[Sequence['SEQCNode']]:
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
    assert min_repetitions_for_for_loop <= min_repetitions_for_shared_wf
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
    __slots__ = ()

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

    def _get_single_indexed_playback(self) -> Optional['WaveformPlayback']:
        """Returns None if there is no or if there are more than one indexed playbacks"""
        # detect if there is only a single indexed playback
        single_indexed_playback = None
        for playback in self.iter_waveform_playbacks():
            if not playback.shared:
                if single_indexed_playback is None:
                    single_indexed_playback = playback
                else:
                    break
        else:
            return single_indexed_playback
        return None

    @abc.abstractmethod
    def _visit_nodes(self, waveform_manager: ProgramWaveformManager):
        """push all concatenated waveforms in the waveform manager"""

    @abc.abstractmethod
    def to_source_code(self, waveform_manager: ProgramWaveformManager, node_name_generator: Iterator[str], line_prefix: str, pos_var_name: str,
                        advance_pos_var: bool = True):
        """besides creating the source code, this function registers all needed waveforms to the program manager
        1. shared waveforms
        2. concatenated waveforms in the correct order

        Args:
            waveform_manager:
            node_name_generator: generates unique names of nodes
            line_prefix:
            pos_var_name:
            advance_pos_var: Indexed playback will not advance the position if set to False. This is used internally
            to optimize repeat statements with a single indexed playback.
        Returns:

        """

    def __eq__(self, other):
        """Compare objects based on __slots__"""
        assert getattr(self, '__dict__', None) is None
        return type(self) == type(other) and all(getattr(self, attr) == getattr(other, attr)
                                                 for base_class in inspect.getmro(type(self))
                                                 for attr in getattr(base_class, '__slots__', ()))


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
        return (type(other) is Scope and
                len(self.nodes) == len(other.nodes) and
                all(n1.same_stepping(n2) for n1, n2 in zip(self.nodes, other.nodes)))

    def _visit_nodes(self, waveform_manager: ProgramWaveformManager):
        for node in self.nodes:
            node._visit_nodes(waveform_manager)

    def to_source_code(self, waveform_manager: ProgramWaveformManager, node_name_generator: Iterator[str],
                       line_prefix: str, pos_var_name: str,
                       advance_pos_var: bool = True):
        for node in self.nodes:
            yield from node.to_source_code(waveform_manager,
                                           line_prefix=line_prefix,
                                           pos_var_name=pos_var_name,
                                           node_name_generator=node_name_generator,
                                           advance_pos_var=advance_pos_var)


class Repeat(SEQCNode):
    """
    stepping: if False resets the pos to initial value after each iteration"""
    __slots__ = ('repetition_count', 'scope')
    INITIAL_POSITION_NAME_TEMPLATE = 'init_pos_{node_name}'

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

    def _visit_nodes(self, waveform_manager: ProgramWaveformManager):
        self.scope._visit_nodes(waveform_manager)

    def _stores_initial_pos(self):
        self_samples = self.samples()
        if self_samples > 0:
            single_playback = self.scope._get_single_indexed_playback()
            if single_playback is None or single_playback.samples() != self_samples:
                # there is more than one indexed playback
                return True
            else:
                # there is only a single indexed playback
                return False
        else:
            # there is no indexed playback
            return False

    def to_source_code(self, waveform_manager: ProgramWaveformManager, node_name_generator: Iterator[str],
                       line_prefix: str, pos_var_name: str, advance_pos_var: bool = True):
        body_prefix = line_prefix + self.INDENTATION

        store_initial_pos = advance_pos_var and self._stores_initial_pos()
        if store_initial_pos:
            node_name = next(node_name_generator)
            initial_position_name = self.INITIAL_POSITION_NAME_TEMPLATE.format(node_name=node_name)

            # store initial position
            yield '{line_prefix}var {init_pos_name} = {pos_var_name};'.format(line_prefix=line_prefix,
                                                                              init_pos_name=initial_position_name,
                                                                              pos_var_name=pos_var_name)

        yield '{line_prefix}repeat({repetition_count}) {{'.format(line_prefix=line_prefix,
                                                                  repetition_count=self.repetition_count)
        if store_initial_pos:
            yield ('{body_prefix}{pos_var_name} = {init_pos_name};'
                   '').format(body_prefix=body_prefix,
                              pos_var_name=pos_var_name,
                              init_pos_name=initial_position_name)
        yield from self.scope.to_source_code(waveform_manager,
                                             line_prefix=body_prefix, pos_var_name=pos_var_name,
                                             node_name_generator=node_name_generator,
                                             advance_pos_var=store_initial_pos)
        yield '{line_prefix}}}'.format(line_prefix=line_prefix)


class SteppingRepeat(SEQCNode):
    STEPPING_REPEAT_COMMENT = ' // stepping repeat'
    __slots__ = ('node_cluster',)

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
                len(self.node_cluster) == len(other.node_cluster) and
                self.node_cluster[0].same_stepping(other.node_cluster[0]))

    def _visit_nodes(self, waveform_manager: ProgramWaveformManager):
        for node in self.node_cluster:
            node._visit_nodes(waveform_manager)

    def to_source_code(self, waveform_manager: ProgramWaveformManager, node_name_generator: Iterator[str],
                       line_prefix: str, pos_var_name: str,
                       advance_pos_var: bool = True):
        body_prefix = line_prefix + self.INDENTATION
        repeat_open = '{line_prefix}repeat({repetition_count}) {{' + self.STEPPING_REPEAT_COMMENT
        yield repeat_open.format(line_prefix=line_prefix,
                                 repetition_count=self.repetition_count)
        yield from self.node_cluster[0].to_source_code(waveform_manager,
                                                       line_prefix=body_prefix, pos_var_name=pos_var_name,
                                                       node_name_generator=node_name_generator,
                                                       advance_pos_var=advance_pos_var)

        # register remaining concatenated waveforms
        for node in itertools.islice(self.node_cluster, 1, None):
            node._visit_nodes(waveform_manager)

        yield '{line_prefix}}}'.format(line_prefix=line_prefix)


class WaveformPlayback(SEQCNode):
    ADVANCE_DISABLED_COMMENT = ' // advance disabled do to parent repetition'

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

    def _visit_nodes(self, waveform_manager: ProgramWaveformManager):
        if not self.shared:
            waveform_manager.request_concatenated(self.waveform)

    def to_source_code(self, waveform_manager: ProgramWaveformManager,
                       node_name_generator: Iterator[str], line_prefix: str, pos_var_name: str,
                       advance_pos_var: bool = True):
        if self.shared:
            yield '{line_prefix}playWave({waveform});'.format(waveform=waveform_manager.request_shared(self.waveform),
                                                              line_prefix=line_prefix)
        else:
            wf_name = waveform_manager.request_concatenated(self.waveform)
            wf_len = len(self.waveform)
            play_cmd = '{line_prefix}playWaveIndexed({wf_name}, {pos_var_name}, {wf_len});'

            if advance_pos_var:
                advance_cmd = ' {pos_var_name} = {pos_var_name} + {wf_len};'
            else:
                advance_cmd = self.ADVANCE_DISABLED_COMMENT
            yield (play_cmd + advance_cmd).format(wf_name=wf_name,
                                                  wf_len=wf_len,
                                                  pos_var_name=pos_var_name,
                                                  line_prefix=line_prefix)
