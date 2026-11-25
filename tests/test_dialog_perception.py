"""Crowded dialog observations, using synthetic UIA trees only."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from core.perception import VisualPerceptionEngine


class Node:
    def __init__(self, role, name='', children=(), offscreen=False):
        self.ControlTypeName = role
        self.Name = name
        self.children = children
        self.IsOffscreen = offscreen
        self.IsEnabled = True
        self.IsPassword = False
        self.AutomationId = ''
        self.NativeWindowHandle = 42
        self.ProcessId = 123
        self.BoundingRectangle = SimpleNamespace(
            left=10, top=10, right=100, bottom=30,
            width=lambda: 90, height=lambda: 20)

    def GetChildren(self):
        return self.children


class DialogPerceptionTests(unittest.TestCase):
    def test_warm_deep_window_uses_one_fresh_scan_not_shallow_plus_deep(self):
        node = Node('EditControl', 'Deep editor')
        for _ in range(18):
            node = Node('PaneControl', children=[node])
        root = Node('WindowControl', 'Editor', [node])
        engine = VisualPerceptionEngine()
        with patch('core.perception.UIA_AVAILABLE', True), patch('core.perception.auto') as auto, \
                patch.object(engine, '_walk_tree', wraps=engine._walk_tree) as walk:
            auto.GetForegroundControl.return_value = root
            first, _ = engine.get_accessibility_tree()
            cold_calls = walk.call_count
            walk.reset_mock()
            second, _ = engine.get_accessibility_tree()
            self.assertEqual(cold_calls, 2)
            self.assertEqual(walk.call_count, 1)
            self.assertEqual([e.name for e in first], [e.name for e in second])
            self.assertEqual(engine._tree_stats['adaptive_depth'], 40)

    def test_focused_editor_survives_truncated_focus_neighborhood(self):
        field = Node('ComboBoxControl', 'Destination')
        form = Node('WindowControl', 'Form', [
            *[Node('PaneControl', children=[Node('ButtonControl', f'Item {i}')]) for i in range(80)], field])
        root = Node('WindowControl', 'Browser', [Node('DocumentControl', 'Page', [form])])
        self.link_parents(root)
        with patch('core.perception.UIA_AVAILABLE', True), patch('core.perception.auto') as auto:
            auto.GetForegroundControl.return_value = root
            auto.GetFocusedControl.return_value = field
            engine = VisualPerceptionEngine()
            elements, _ = engine.get_accessibility_tree()
        self.assertTrue(engine._tree_stats['focus_truncated'])
        self.assertTrue(engine._tree_stats['focused_editor_captured'])
        self.assertEqual(elements[0].name, 'Destination')
        self.assertEqual(elements[0].runtime_id, (id(field),))
        self.assertLessEqual(len(elements), 40)

    def test_focused_editor_outside_window_is_rejected(self):
        root = Node('WindowControl', 'Browser')
        other = Node('WindowControl', 'Other', [Node('EditControl', 'Private')])
        self.link_parents(root)
        self.link_parents(other)
        with patch('core.perception.auto') as auto:
            auto.GetFocusedControl.return_value = other.children[0]
            self.assertIsNone(VisualPerceptionEngine._focused_editor(root))

    def test_late_page_editor_is_selected_before_button_budget(self):
        page = Node('DocumentControl', 'Page', [
            *[Node('ButtonControl', f'Action {i}') for i in range(35)], Node('EditControl', 'Input')])
        root = Node('WindowControl', 'Browser', [page])
        self.assertIn('Input', [e.name for e in self.observe(root)])

    @staticmethod
    def link_parents(node, parent=None):
        node.GetRuntimeId = lambda: (id(node),)
        node.GetParentControl = lambda: parent
        for child in node.children:
            DialogPerceptionTests.link_parents(child, node)

    def test_truncated_page_recovers_focused_form_without_scanning_entire_inbox(self):
        field = Node('EditControl', 'Destination')
        form = Node('WindowControl', 'New item', [field, Node('EditControl', 'Title'),
                                                Node('EditControl', 'Content')])
        page = Node('DocumentControl', 'Page', [
            *[Node('PaneControl', children=[Node('ButtonControl', f'Row {i}')]) for i in range(80)], form])
        root = Node('WindowControl', 'Browser', [Node('EditControl', 'Address'), page])
        self.link_parents(root)
        with patch('core.perception.UIA_AVAILABLE', True), patch('core.perception.auto') as auto:
            auto.GetForegroundControl.return_value = root
            auto.GetFocusedControl.return_value = field
            engine = VisualPerceptionEngine()
            elements, _ = engine.get_accessibility_tree()
        for name in ('Destination', 'Title', 'Content', 'Address'):
            self.assertIn(name, [e.name for e in elements])
        self.assertTrue(engine._tree_stats['focus_retry'])
        self.assertLessEqual(engine._tree_stats['focus_visited'], 256)
        self.assertLessEqual(len(elements), 40)
        self.assertEqual([e.tag for e in elements], list(range(1, len(elements)+1)))

    def test_focus_in_another_window_is_never_merged(self):
        root = Node('WindowControl', 'Browser')
        outside = Node('WindowControl', 'Other app', [Node('EditControl', 'Private field')])
        self.link_parents(root)
        self.link_parents(outside)
        with patch('core.perception.auto') as auto:
            auto.GetFocusedControl.return_value = outside.children[0]
            self.assertIsNone(VisualPerceptionEngine._focused_region(root))

    def test_cyclic_focus_ancestry_is_bounded(self):
        root = Node('WindowControl', 'Browser')
        self.link_parents(root)
        field = Node('EditControl', 'Field')
        field.GetRuntimeId = lambda: (123,)
        field.GetParentControl = lambda: field
        with patch('core.perception.auto') as auto:
            auto.GetFocusedControl.return_value = field
            self.assertIsNone(VisualPerceptionEngine._focused_region(root))

    def test_address_bar_does_not_suppress_deep_page_retry(self):
        page = Node('ButtonControl', 'Create item')
        for _ in range(18):
            page = Node('PaneControl', children=[page])
        root = Node('WindowControl', 'Browser', [
            Node('EditControl', 'Address'), Node('DocumentControl', 'Page', [page])])
        names = [e.name for e in self.observe(root)]
        self.assertIn('Create item', names)
        self.assertIn('Address', names)

    def test_native_toolbar_does_not_hide_page_actions(self):
        root = Node('WindowControl', 'Browser', [
            Node('PaneControl', children=[Node('ButtonControl', f'Tool {i}') for i in range(40)]),
            Node('EditControl', 'Address'),
            Node('DocumentControl', 'Page', [Node('ButtonControl', 'Create item'),
                 Node('EditControl', 'Title'), Node('ButtonControl', 'Save draft')])])
        elements = self.observe(root)
        names = [e.name for e in elements]
        for name in ('Create item', 'Title', 'Save draft', 'Address'):
            self.assertIn(name, names)
        self.assertLessEqual(len(elements), 40)

    def test_toolbar_cannot_exhaust_budget_before_deep_editor(self):
        editor = Node('EditControl', 'Editor content')
        for _ in range(20):
            editor = Node('PaneControl', children=[editor])
        root = Node('WindowControl', 'Visual Studio Code', [
            Node('PaneControl', children=[Node('ButtonControl', f'Tool {i}') for i in range(40)]),
            editor])
        elements = self.observe(root)
        self.assertEqual(len(elements), 40)
        self.assertIn('Editor content', [e.name for e in elements])
        self.assertEqual([e.tag for e in elements], list(range(1, 41)))

    def test_deep_editor_is_found_by_bounded_retry(self):
        node = Node('EditControl', 'Editor content')
        for _ in range(20):
            node = Node('PaneControl', children=[node])
        root = Node('WindowControl', 'Visual Studio Code', [node])
        with patch('core.perception.UIA_AVAILABLE', True), patch('core.perception.auto') as auto:
            auto.GetForegroundControl.return_value = root
            engine = VisualPerceptionEngine()
            elements, _ = engine.get_accessibility_tree()
        self.assertIn('Editor content', [e.name for e in elements])
        self.assertTrue(engine._tree_stats['deep_retry'])
        self.assertEqual(engine._tree_stats['depth_cutoffs'], 0)

    def test_depth_limit_remains_bounded_and_reported(self):
        node = Node('EditControl', 'Too deep')
        for _ in range(70):
            node = Node('PaneControl', children=[node])
        with patch('core.perception.UIA_AVAILABLE', True), patch('core.perception.auto') as auto:
            auto.GetForegroundControl.return_value = node
            engine = VisualPerceptionEngine()
            engine.get_accessibility_tree()
        self.assertGreater(engine._tree_stats['depth_cutoffs'], 0)
        self.assertLessEqual(engine._tree_stats['visited'], 512)

    def test_unavailable_provider_and_shallow_tree_are_not_deep_retried(self):
        with patch('core.perception.UIA_AVAILABLE', True), patch('core.perception.auto') as auto:
            auto.GetForegroundControl.return_value = Node('WindowControl', children=[Node('ButtonControl', 'Close')])
            engine = VisualPerceptionEngine()
            engine.get_accessibility_tree()
            self.assertFalse(engine._tree_stats['deep_retry'])
        with patch('core.perception.UIA_AVAILABLE', False):
            engine.get_accessibility_tree()
            self.assertEqual(engine.accessibility_diagnostic(), 'Accessibility scan unavailable.')

    def observe(self, root, budget=40):
        with patch('core.perception.UIA_AVAILABLE', True), patch('core.perception.auto') as auto:
            auto.GetForegroundControl.return_value = root
            return VisualPerceptionEngine(max_elements=budget).get_accessibility_tree()[0]

    def test_crowded_file_list_does_not_hide_filename_and_save(self):
        rows = [Node('ListItemControl', f'fixture-{i}.txt', [
            Node('EditControl', 'Name'), Node('EditControl', 'Date modified'),
            Node('EditControl', 'Type'), Node('EditControl', 'Size')]) for i in range(1000)]
        root = Node('WindowControl', 'Save as', [
            Node('PaneControl', children=[Node('ListControl', children=rows)]),
            Node('PaneControl', children=[Node('ComboBoxControl', 'File name:'),
                Node('ButtonControl', 'Save'), Node('ButtonControl', 'Cancel')])])
        elements = self.observe(root)
        names = [e.name for e in elements]
        self.assertIn('File name:', names)
        self.assertIn('Save', names)
        self.assertIn('Cancel', names)
        self.assertIn('fixture-0.txt', names)
        self.assertLessEqual(len(elements), 40)
        self.assertEqual([e.tag for e in elements], list(range(1, len(elements) + 1)))

    def test_offscreen_controls_are_not_action_targets(self):
        elements = self.observe(Node('WindowControl', children=[
            Node('ButtonControl', 'Hidden', offscreen=True), Node('ButtonControl', 'Save')]))
        self.assertEqual([e.name for e in elements], ['Save'])

    def test_calculator_display_survives_normal_button_tree(self):
        elements = self.observe(Node('WindowControl', 'Calculator', [
            Node('PaneControl', children=[Node('ButtonControl', str(i)) for i in range(10)]),
            Node('TextControl', 'Display is 42')]))
        self.assertIn('Display is 42', [e.name for e in elements])


if __name__ == '__main__':
    unittest.main()
