from typing import Iterable, Union, List, Generator, Tuple
from collections import deque, namedtuple
from copy import copy as shallow_copy
import weakref


__all__ = ['Node']


def make_empty_weak_reference() -> weakref.ref:
    return weakref.ref(lambda: None)


class Node:
    def __init__(self, parent: Union['Node', None]=None, children: Iterable=list()):
        self.__parent = make_empty_weak_reference() if parent is None else weakref.ref(parent)
        self.__children = [self.parse_child(child) for child in children]

    def parse_child(self, child) -> 'Node':
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

    def __iter__(self) -> Iterable['Node']:
        return iter(self.children)

    def __setitem__(self, idx: Union[int, slice], value: Union['Node', Iterable['Node']]):
        if isinstance(idx, slice):
            if isinstance(value, Node):
                raise TypeError('can only assign an iterable (Loop does not count)')
            value = (self.parse_child(child) for child in value)
        else:
            value = self.parse_child(value)
        self.__children.__setitem__(idx, value)

    def __getitem__(self, *args, **kwargs) ->Union['Node', List['Node']]:
        return self.__children.__getitem__(*args, **kwargs)

    def __len__(self) -> int:
        return len(self.__children)

    def get_depth_first_iterator(self) -> Generator['Node', None, None]:
        if not self.is_leaf():
            for e in self.__children:
                yield from e.get_depth_first_iterator()
        yield self

    def get_breadth_first_iterator(self, queue=deque()) -> Generator['Node', None, None]:
        yield self
        if not self.is_leaf():
            queue.extend(self.__children)
        if queue:
            yield from queue.popleft().get_breadth_first_iterator(queue)

    def assert_tree_integrity(self) -> None:
        for child in self.__children:
            if id(child.parent) != id(self):
                raise AssertionError('Child is missing parent reference')
            child.assert_tree_integrity()
        if self.parent:
            if id(self) not in (id(c) for c in self.parent.__children):
                raise AssertionError('Parent is missing child reference')

    @property
    def children(self) -> List['Node']:
        """
        :return: shallow copy of children
        """
        return shallow_copy(self.__children)

    @property
    def parent(self) -> Union[None, 'Node']:
        return self.__parent()

    def get_root(self) -> 'Node':
        if self.parent:
            return self.parent.get_root()
        else:
            return self

    def get_location(self) -> Tuple[int, ...]:
        if self.parent:
            for i, c in enumerate(self.parent.__children):
                if id(c) == id(self):
                    return (*self.parent.get_location(), i)
            raise AssertionError('Self not found in parent')
        else:
            return tuple()

    def locate(self, location: Tuple[int, ...]):
        if location:
            return self.__children[location[0]].locate(location[1:])
        else:
            return self


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
