import unittest
from pathlib import Path
from core.agent_checks import document_prerequisite, misplaced_document_edit
from core.perception import ScreenState, UIElement
from core.plugins import PluginRegistry
from core.reasoning import NextAction


class DocumentPrerequisiteTests(unittest.TestCase):
    goal = r'In Notepad open C:\Synthetic fixtures\source.txt. Replace PENDING with READY. Save as C:\Synthetic fixtures\reviewed.txt.'

    def test_wrong_tab_does_not_receive_replace_recipe(self):
        registry = PluginRegistry(Path(__file__).resolve().parents[1] / 'data/plugins')
        registry.discover()
        hint = registry.hint_for_prompt('*Unrelated draft - Notepad', self.goal)
        self.assertIn('DOCUMENT PREREQUISITE', hint)
        self.assertNotIn('ctrl+h', hint)
        self.assertIn('ctrl+h', registry.hint_for_prompt('source.txt - Notepad', self.goal))

    def test_edit_actions_blocked_but_open_navigation_remains_available(self):
        state = ScreenState('', '', [UIElement(1, 'DocumentControl', 'Text editor', (0,0,20,20), True),
                                    UIElement(2, 'ButtonControl', 'Replace all', (20,0,40,20), True)],
                            '*Other - Notepad', 100, 100)
        for action in (NextAction('key', text='ctrl+h'), NextAction('type', target_tag=1, text='READY'),
                       NextAction('click', target_tag=2), NextAction('key', text='ctrl+s')):
            self.assertTrue(misplaced_document_edit(state, action, self.goal))
        self.assertFalse(misplaced_document_edit(state, NextAction('key', text='ctrl+o'), self.goal))
        self.assertFalse(misplaced_document_edit(state, NextAction('key', text='escape'), self.goal))
        state.foreground_window_title = 'Open'
        self.assertFalse(misplaced_document_edit(state, NextAction('type', text='source.txt'), self.goal))

    def test_matched_document_and_goals_without_explicit_path_are_unchanged(self):
        for title in ('source.txt - Notepad', '*SOURCE.TXT - Notepad', 'source - Notepad'):
            self.assertEqual(document_prerequisite(title, self.goal), '')
        self.assertEqual(document_prerequisite('Other - Notepad', 'Replace PENDING with READY in the current document'), '')
        self.assertEqual(document_prerequisite('file.py - Visual Studio Code', self.goal), '')


if __name__ == '__main__':
    unittest.main()
