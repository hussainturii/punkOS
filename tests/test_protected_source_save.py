import unittest
from unittest.mock import Mock, patch

from core.agent_checks import protected_source_save, open_file_dialog
from core.perception import ScreenState, UIElement
from core.reasoning import NextAction
from core.safety import SafetySystem


class ProtectedSourceTests(unittest.TestCase):
    goal = r'Open Notepad, then open C:\Fixtures\source.txt. Replace PENDING with READY. Save as C:\Fixtures\reviewed.txt, leaving source.txt unchanged.'

    def state(self, title, elements=None):
        return ScreenState('', '', elements or [], title, 100, 100)

    def test_planner_never_executes_protected_source_save(self):
        from core.planner import StrategicPlanner
        perception, reasoning, controller = Mock(), Mock(), Mock()
        perception.perceive.return_value = self.state('*source.txt - Notepad')
        reasoning.decide_next_action.return_value = NextAction('key', text='ctrl+s')
        reasoning.recover.return_value = NextAction('fail', text='Stopped')
        planner = StrategicPlanner(perception, reasoning, controller, SafetySystem([]),
                                   use_planning_phase=False, max_steps=1, step_delay=0,
                                   on_log=lambda _: None)
        with patch('core.document_evidence.DocumentEvidence.source_prerequisite', return_value=''):
            result = planner.run(self.goal)
        self.assertFalse(result.success)
        controller.execute.assert_not_called()
        self.assertIn('original source must remain unchanged', result.history[-1]['outcome'])

    def test_source_save_blocked_but_save_as_and_output_save_allowed(self):
        for title in ('source.txt - Notepad', '*SOURCE.TXT - Notepad', '*source - Notepad'):
            state = self.state(title)
            self.assertTrue(protected_source_save(state, NextAction('key', text='ctrl+s'), self.goal))
            self.assertFalse(protected_source_save(state, NextAction('key', text='ctrl+shift+s'), self.goal))
        self.assertFalse(protected_source_save(self.state('reviewed.txt - Notepad'), NextAction('key', text='ctrl+s'), self.goal))

    def test_menu_save_and_ordinary_edit(self):
        state = self.state('*source.txt - Notepad', [UIElement(1, 'MenuItemControl', '&Save\tCtrl+S', (0,0,10,10), True)])
        self.assertTrue(protected_source_save(state, NextAction('click', target_tag=1), self.goal))
        self.assertFalse(protected_source_save(state, NextAction('key', text='ctrl+s'), r'Open C:\Fixtures\source.txt and update it.'))

    def test_parent_titled_open_picker_ignores_path_keywords(self):
        elements = [UIElement(i, role, name, (0,0,10,10), True) for i, role, name in
                    [(1, 'EditControl', 'File name:'), (2, 'ButtonControl', 'Open'), (3, 'ButtonControl', 'Cancel')]]
        state = self.state('backup-note.txt - Notepad', elements)
        action = NextAction('type', target_tag=1, text=r'C:\Fixtures\replace\source.txt')
        self.assertTrue(open_file_dialog(state))
        self.assertFalse(SafetySystem([]).assess(action, screen_state=state).requires_confirmation)
        state.elements.append(UIElement(4, 'ButtonControl', 'Save', (0,0,10,10), True))
        self.assertFalse(open_file_dialog(state))
        self.assertTrue(SafetySystem([]).assess(action, screen_state=state).requires_confirmation)

    def test_partial_and_unrelated_dialogs_not_exempt(self):
        state = self.state('Open', [UIElement(1, 'EditControl', 'File name:', (0,0,10,10), True)])
        self.assertFalse(open_file_dialog(state))
        state.elements.extend([UIElement(2, 'ButtonControl', 'Open', (0,0,10,10), True), UIElement(3, 'ButtonControl', 'Cancel', (0,0,10,10), True)])
        state.foreground_window_title = 'Save as'
        self.assertFalse(open_file_dialog(state))


if __name__ == '__main__':
    unittest.main()
