import copy
import unittest
from pathlib import Path

from core.plugins import PluginRegistry


class PluginRoutingTests(unittest.TestCase):
    def setUp(self):
        self.registry = PluginRegistry(Path(__file__).resolve().parents[1] / 'data/plugins')
        self.registry.discover()

    def test_notepad_does_not_borrow_vscode_open_shortcut(self):
        self.assertIsNone(self.registry.find_workflow('open file utils.py', 'Untitled - Notepad'))
        hint = self.registry.hint_for_prompt('Untitled - Notepad', 'open file utils.py')
        self.assertIn('Notepad', hint)
        self.assertNotIn('ctrl+p', hint)
        self.assertNotIn('Visual Studio Code', hint)

    def test_unknown_or_missing_window_cannot_trigger_app_shortcuts(self):
        for window in ('Calculator', 'PunkOS', 'Save As', ''):
            for goal in ('open file utils.py', 'create table', 'write and save', 'replace foo'):
                with self.subTest(window=window, goal=goal):
                    self.assertIsNone(self.registry.find_workflow(goal, window))
                    self.assertEqual(self.registry.hint_for_prompt(window, goal), '')

    def test_correct_app_workflows_still_match(self):
        for window, goal, plugin_id, shortcut in (
            ('main.py - Visual Studio Code', 'open file utils.py', 'vscode', 'ctrl+p'),
            ('Untitled - Notepad', 'write and save', 'notepad', 'ctrl+s'),
            ('Book1 - Excel', 'go to cell A1', 'excel', 'ctrl+g'),
        ):
            with self.subTest(plugin=plugin_id):
                match = self.registry.find_workflow(goal, window)
                self.assertIsNotNone(match)
                self.assertEqual(match[0].plugin_id, plugin_id)
                self.assertIn(shortcut, self.registry.hint_for_prompt(window, goal))

    def test_disabled_plugin_cannot_supply_workflows(self):
        self.registry.plugins['vscode'].is_active = False
        self.assertEqual(self.registry.hint_for_prompt('main.py - Visual Studio Code', 'open file utils.py'), '')

    def test_overlapping_window_matches_abstain_independent_of_order(self):
        duplicate = copy.deepcopy(self.registry.plugins['notepad'])
        duplicate.plugin_id = 'other_editor'
        self.registry.plugins[duplicate.plugin_id] = duplicate
        for reverse in (False, True):
            if reverse:
                self.registry.plugins = dict(reversed(list(self.registry.plugins.items())))
            self.assertIsNone(self.registry.plugin_for_window('Untitled - Notepad'))
            self.assertEqual(self.registry.hint_for_prompt('Untitled - Notepad', 'write and save'), '')


if __name__ == '__main__':
    unittest.main()
