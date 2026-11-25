import unittest
from unittest.mock import patch, Mock

from core.controller import DesktopController
from core.perception import ScreenState, UIElement
from core.reasoning import NextAction
from core.skills import SkillLibrary


class DesktopInputTests(unittest.TestCase):
    @patch('core.controller.pyautogui')
    def test_focus_change_blocks_input(self, keyboard):
        from core.window_handoff import Window
        backend = Mock()
        backend.foreground.return_value = Window(88, 'Another window', 'other.exe')
        self.calc.foreground_window_handle = 42
        result = DesktopController(backend).execute(NextAction('type', text='27'), self.calc)
        self.assertFalse(result.success)
        self.assertEqual(keyboard.mock_calls, [])

    def test_confirmation_restores_exact_window(self):
        from core.window_handoff import Window
        backend = Mock()
        backend.foreground.return_value = Window(42, 'Calculator', 'calculator.exe')
        self.calc.foreground_window_handle = 42
        self.assertTrue(DesktopController(backend).restore_focus(self.calc))
        backend.activate.assert_called_once_with(42)

    @patch('core.controller.pyautogui')
    def test_confirmation_focus_failures_are_diagnostic_and_send_no_input(self, keyboard):
        from core.window_handoff import Window
        backend = Mock()
        controller = DesktopController(backend)
        self.calc.foreground_window_handle = 42
        backend.foreground.return_value = Window(88, 'Private title', 'other.exe')
        self.assertFalse(controller.restore_focus(self.calc))
        self.assertIn('foreground handle is 88', controller.focus_restore_detail)
        self.assertNotIn('Private', controller.focus_restore_detail)
        backend.foreground.return_value = None
        self.assertFalse(controller.restore_focus(self.calc))
        self.assertIn('No foreground', controller.focus_restore_detail)
        backend.activate.side_effect = OSError('private details')
        self.assertFalse(controller.restore_focus(self.calc))
        self.assertIn('OSError', controller.focus_restore_detail)
        self.assertNotIn('private', controller.focus_restore_detail)
        self.assertEqual(keyboard.mock_calls, [])

    def test_confirmation_without_observed_handle_does_not_activate(self):
        backend = Mock()
        controller = DesktopController(backend)
        self.calc.foreground_window_handle = 0
        self.assertFalse(controller.restore_focus(self.calc))
        backend.activate.assert_not_called()
        self.assertIn('no window handle', controller.focus_restore_detail)

    def setUp(self):
        self.controller = DesktopController()
        self.calc = ScreenState('', '', [UIElement(i, 'ButtonControl', name, (0,0,20,20), True)
            for i, name in enumerate(('Equals', 'Plus', 'Multiply by'), 1)], 'Calculator', 100, 100)

    @patch('core.controller.pyautogui')
    def test_calculator_uses_number_keys_not_paste(self, keyboard):
        result = self.controller.execute(NextAction('type', text='27'), self.calc)
        self.assertTrue(result.success)
        keyboard.write.assert_called_once_with('27', interval=0.05)
        keyboard.hotkey.assert_not_called()
        self.assertIn('must be verified', result.message)

    @patch('core.controller.pyautogui')
    def test_calculator_expression_rejected_before_input(self, keyboard):
        result = self.controller.execute(NextAction('type', text='27 * 14 + 125 ='), self.calc)
        self.assertFalse(result.success)
        self.assertEqual(keyboard.mock_calls, [])

    @patch('core.controller.pyautogui')
    def test_invalid_target_never_types_into_arbitrary_focus(self, keyboard):
        for tag in (1, 99):
            state = ScreenState('', '', [UIElement(1, 'ButtonControl', 'Submit', (0,0,20,20), True)], 'Editor', 100, 100)
            self.assertFalse(self.controller.execute(NextAction('type', target_tag=tag, text='hello'), state).success)
        self.assertEqual(keyboard.mock_calls, [])

    def test_calculator_skill_is_scoped(self):
        self.assertIn('SKILL calculator:', SkillLibrary().context('Calculate', 'Calculator'))
        self.assertNotIn('SKILL calculator:', SkillLibrary().context('Edit a note', 'Notepad'))


if __name__ == '__main__':
    unittest.main()
