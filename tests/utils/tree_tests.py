import unittest

from qctoolkit.utils.tree import Node


class SpecialNode(Node):
    def __init__(self, my_argument=None, **kwargs):
        super().__init__(**kwargs)
        self.init_arg = my_argument


class NodeTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_init(self):

        children = [Node(parent=None, children=[]) for _ in range(5)]

        root = Node(parent=None, children=children)

        for c1, c2 in zip(children, root):
            self.assertIs(c1, c2)
            self.assertIs(c1.parent, root)

    def test_parse_children(self):

        root = Node()
        to_parse = Node()
        parsed = root.parse_child(to_parse)
        self.assertIs(parsed.parent, root)
        # maybe change this behaviour?
        self.assertIs(parsed, to_parse)

        sub_node = Node()
        to_parse = dict(children=[sub_node])
        parsed = root.parse_child(to_parse)
        self.assertIs(parsed.parent, root)
        self.assertIs(parsed.children[0], sub_node)
        with self.assertRaises(TypeError):
            root.parse_child(SpecialNode())

    def test_parse_children_derived(self):
        root = SpecialNode()
        to_parse = SpecialNode()
        parsed = root.parse_child(to_parse)
        self.assertIs(parsed.parent, root)
        # maybe change this behaviour?
        self.assertIs(parsed, to_parse)

        sub_node = SpecialNode()
        to_parse = dict(children=[sub_node], my_argument=6)
        parsed = root.parse_child(to_parse)
        self.assertIs(parsed.parent, root)
        self.assertEqual(parsed.init_arg, 6)
        with self.assertRaises(TypeError):
            root.parse_child(Node())

    def test_set_item(self):
        root = SpecialNode()

        with self.assertRaises(TypeError):
            root[:] = SpecialNode()
        to_insert = [SpecialNode(), SpecialNode()]

        root[:] = to_insert
        for c, e in zip(root, to_insert):
            self.assertIs(c, e)
            self.assertIs(c.parent, root)

        to_overwrite = SpecialNode()
        root[1] = to_overwrite
        self.assertIs(root[0], to_insert[0])
        self.assertIs(root[1], to_overwrite)

    def test_assert_integrity(self):
        root = Node(children=(Node(), Node()))

        root.assert_tree_integrity()

        root_children = getattr(root, '_Node__children')

        root_children[1] = Node()

        with self.assertRaises(AssertionError):
            root.assert_tree_integrity()

    def test_depth_iteration(self):
        root = Node(children=[Node(children=[Node(), Node()]), Node()])

        depth_nodes = tuple(root.get_depth_first_iterator())

        self.assertEqual(depth_nodes, (root[0][0], root[0][1], root[0], root[1], root))

    def test_breadth_iteration(self):
        root = Node(children=[Node(children=[Node(), Node()]), Node()])

        breadth_nodes = tuple(root.get_breadth_first_iterator())

        self.assertEqual(breadth_nodes, (root, root[0], root[1], root[0][0], root[0][1]))

