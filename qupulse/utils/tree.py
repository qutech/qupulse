"""This module contains a tree implementation."""

from typing import Iterable, Union, List, Generator, Tuple, TypeVar, Optional, Sequence
from collections import deque, namedtuple
import weakref

from qupulse.utils.types import SequenceProxy


__all__ = ['Node']


def make_empty_weak_reference() -> weakref.ref:
    return weakref.ref(lambda: None)


_NodeType = TypeVar('_NodeType', bound='Node')


class Node:
    debug = False
    __slots__ = ('__parent', '__children', '__parent_index', '__weakref__')

    def __init__(self: _NodeType,
                 parent: Union[_NodeType, None]=None,
                 children: Optional[Iterable]=None):
        self.__parent = make_empty_weak_reference() if parent is None else weakref.ref(parent)
        self.__children = [] if children is None else [self.parse_child(child) for child in children]
        self.__parent_index = None

        for i, child in enumerate(self.__children):
            self.__children[i].__parent_index = i

    def parse_child(self: _NodeType, child) -> _NodeType:
        if isinstance(child, dict):
            return type(self)(parent=self, **child)
        elif type(child) is type(self):
            child.__parent = weakref.ref(self)
            return child
        else:
            raise TypeError('Invalid child type', type(child))

    def is_leaf(self) -> bool:
        return len(self.__children) == 0

    def depth(self) -> int:
        return 0 if self.is_leaf() else (1 + max(e.depth() for e in self.__children))

    def is_balanced(self) -> bool:
        if self.is_leaf():
            return True
        return all((e.depth() == self.__children[0].depth() and e.is_balanced()) for e in self.__children)

    def __iter__(self: _NodeType) -> Iterable[_NodeType]:
        return iter(self.__children)

    def __reversed__(self: _NodeType) -> Iterable[_NodeType]:
        return reversed(self.__children)

    def __setitem__(self: _NodeType, idx: Union[int, slice], value: Union[_NodeType, Iterable[_NodeType]]):
        if isinstance(idx, slice):
            if isinstance(value, Node):
                raise TypeError('can only assign an iterable (Loop does not count)')
            value = tuple(self.parse_child(child) for child in value)
            indices = range(*idx.indices(len(self.__children)))
            self.__children.__setitem__(idx, value)

            if len(value) != len(indices):
                first_invalid = indices.start if indices.step > 0 else indices.stop
                for index in range(first_invalid, len(self)):
                    self.__children[index].__parent_index = index
            elif len(value) > 0:
                for index in range(indices.start, indices.start + indices.step*len(value)):
                    self.__children[index].__parent_index = index

        else:
            value = self.parse_child(value)
            value.__parent_index = idx
            self.__children.__setitem__(idx, value)

    def __getitem__(self: _NodeType, *args, **kwargs) ->Union[_NodeType, List[_NodeType]]:
        return self.__children.__getitem__(*args, **kwargs)

    def __len__(self) -> int:
        return len(self.__children)

    def get_depth_first_iterator(self: _NodeType) -> Generator[_NodeType, None, None]:
        stack = [(self, self.__children)]

        while stack:
            node, children = stack.pop()

            if children:
                stack.append((node, None))
                stack.extend((child, child.__children) for child in reversed(children))
            else:
                yield node

    def get_breadth_first_iterator(self: _NodeType) -> Generator[_NodeType, None, None]:
        queue = deque([self])
        while queue:
            elem = queue.popleft()
            queue.extend(elem)
            yield elem

    def assert_tree_integrity(self) -> None:
        if self.debug:
            for child in self.__children:
                if id(child.parent) != id(self):
                    raise AssertionError('Child is missing parent reference')
                child.assert_tree_integrity()
            if self.parent:
                if self.__parent_index not in range(len(self.parent)):
                    raise AssertionError('Out of range parent index')
                if id(self.parent[self.__parent_index]) != id(self):
                    if id(self) in (id(c) for c in self.parent.__children):
                        raise AssertionError('Wrong parent index')
                    else:
                        raise AssertionError('Parent is missing child reference')

    @property
    def children(self: _NodeType) -> Sequence[_NodeType]:
        """
        :return: shallow copy of children
        """
        return SequenceProxy(self.__children)

    @property
    def parent(self: _NodeType) -> Union[None, _NodeType]:
        return self.__parent()

    @property
    def parent_index(self) -> int:
        return self.__parent_index

    def get_root(self: _NodeType) -> _NodeType:
        if self.parent:
            return self.parent.get_root()
        else:
            return self

    def get_location(self) -> Tuple[int, ...]:
        self.assert_tree_integrity()
        if self.parent:
            return (*self.parent.get_location(), self.__parent_index)
        else:
            return tuple()

    def locate(self: _NodeType, location: Tuple[int, ...]) -> _NodeType:
        if location:
            return self.__children[location[0]].locate(location[1:])
        else:
            return self

    def _reverse_children(self):
        """Reverse children in-place"""
        self.__children.reverse()


def is_tree_circular(root: Node) -> Union[None, Tuple[List[Node], int]]:
    NodeStack = namedtuple('NodeStack', ['node', 'stack'])

    nodes_to_visit = deque((NodeStack(root, deque()), ))

    while nodes_to_visit:
        node, stack = nodes_to_visit.pop()

        stack.append(id(node))
        for child in node:
            if id(child) in stack:
                return stack, id(child)

            nodes_to_visit.append((child, stack))
    return None
