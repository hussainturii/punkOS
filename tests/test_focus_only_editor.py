import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from core.perception import UIElement, ScreenState, VisualPerceptionEngine
from core.controller import DesktopController
from core.reasoning import NextAction
from tests.test_dialog_perception import Node


class FocusOnlyEditorTests(unittest.TestCase):
    def test_focus_lost_after_selection_prevents_text_delivery(self):
        for role in ('ComboBoxControl', 'EditControl'):
            with self.subTest(role=role):
                element = UIElement(1, role, 'Input', (10,10,100,30), True, runtime_id=(7,))
                state = ScreenState('', '', [element], 'Form', 100, 100)
                clipboard = Mock()
                clipboard.paste.return_value = 'previous'
                with patch.dict('sys.modules', {'pyperclip': clipboard}), patch('core.controller.time.sleep'), \
                        patch('core.controller.pyautogui') as keyboard, \
                        patch.object(DesktopController, '_typing_focus_matches', side_effect=[True, False]):
                    result = DesktopController()._type(NextAction('type', target_tag=1, text='sample'), state)
                self.assertFalse(result.success)
                self.assertFalse(result.input_unverified)
                self.assertIn('focus changed', result.message)
                keyboard.write.assert_not_called()
                self.assertEqual([c.args for c in keyboard.hotkey.call_args_list],
                                 [('ctrl', 'a')] if role == 'EditControl' else [])
                if role == 'EditControl':
                    clipboard.copy.assert_any_call('previous')

    @patch('core.controller.pyautogui')
    def test_url_newline_is_rejected_before_input_and_does_not_latch_guard(self, keyboard):
        from core.input_evidence import InputEvidence
        element = UIElement(1, 'EditControl', 'Address and search bar', (10,10,100,30), True, runtime_id=(7,))
        state = ScreenState('', '', [element], 'Browser', 100, 100)
        for value in ('example.test\n', 'https://example.test/path\r\n'):
            action = NextAction('type', target_tag=1, text=value)
            result = DesktopController()._type(action, state)
            self.assertFalse(result.success)
            self.assertFalse(result.input_unverified)
            self.assertIn('separate Enter', result.message)
            guard = InputEvidence()
            guard.record(action, state, result)
            self.assertFalse(guard.restriction(NextAction('type', target_tag=1, text=value.strip())))
        keyboard.click.assert_not_called()
        keyboard.hotkey.assert_not_called()
        keyboard.write.assert_not_called()

    def test_document_url_line_ending_is_not_silently_changed_or_rejected(self):
        value = 'https://example.test/path\n'
        element = UIElement(1, 'EditControl', 'Document content', (10,10,100,30), True, runtime_id=(7,))
        state = ScreenState('', '', [element], 'Editor', 100, 100)
        clipboard = Mock()
        clipboard.paste.return_value = 'previous'
        with patch.dict('sys.modules', {'pyperclip': clipboard}), patch('core.controller.time.sleep'), \
                patch('core.controller.pyautogui'), \
                patch.object(DesktopController, '_typing_focus_matches', return_value=True), \
                patch.object(DesktopController, '_focused_text', return_value=value):
            result = DesktopController()._type(NextAction('type', target_tag=1, text=value), state)
        self.assertTrue(result.success)
        clipboard.copy.assert_any_call(value)

    def test_short_ascii_combo_uses_one_keyboard_entry_without_clipboard(self):
        element = UIElement(1, 'ComboBoxControl', 'Choice', (10,10,100,30), True, runtime_id=(7,))
        state = ScreenState('', '', [element], 'Form', 100, 100)
        clipboard = Mock()
        with patch.dict('sys.modules', {'pyperclip': clipboard}), patch('core.controller.time.sleep'), \
                patch('core.controller.pyautogui') as keyboard, \
                patch.object(DesktopController, '_typing_focus_matches', return_value=True), \
                patch.object(DesktopController, '_focused_text', return_value=None):
            result = DesktopController()._type(NextAction('type', target_tag=1, text='choice-42'), state)
        keyboard.write.assert_called_once_with('choice-42', interval=.03)
        self.assertFalse(any(c.args == ('ctrl', 'v') for c in keyboard.hotkey.call_args_list))
        clipboard.copy.assert_not_called()
        self.assertTrue(result.input_unverified)
        self.assertIn('keyboard entry', result.message)

    def test_empty_combo_does_not_trigger_select_all_focus_transition(self):
        element = UIElement(1, 'ComboBoxControl', 'Entry', (10,10,100,30), True, runtime_id=(7,))
        state = ScreenState('', '', [element], 'Form', 100, 100)
        focused = [True]
        def shortcut(*keys):
            if keys == ('ctrl', 'a'):
                focused[0] = False
        with patch('core.controller.time.sleep'), patch('core.controller.pyautogui') as keyboard, \
                patch.object(DesktopController, '_typing_focus_matches', side_effect=lambda _: focused[0]), \
                patch.object(DesktopController, '_focused_text', return_value='sample'):
            keyboard.hotkey.side_effect = shortcut
            result = DesktopController()._type(NextAction('type', target_tag=1, text='sample'), state)
        self.assertTrue(result.success)
        keyboard.hotkey.assert_not_called()
        keyboard.write.assert_called_once_with('sample', interval=.03)

    def test_known_combo_text_keeps_replacement_behavior(self):
        element = UIElement(1, 'ComboBoxControl', 'Filename', (10,10,100,30), True,
                            text='old.txt', runtime_id=(7,))
        state = ScreenState('', '', [element], 'Form', 100, 100)
        with patch('core.controller.time.sleep'), patch('core.controller.pyautogui') as keyboard, \
                patch.object(DesktopController, '_typing_focus_matches', return_value=True), \
                patch.object(DesktopController, '_focused_text', return_value='new.txt'):
            result = DesktopController()._type(NextAction('type', target_tag=1, text='new.txt'), state)
        self.assertTrue(result.success)
        keyboard.hotkey.assert_called_once_with('ctrl', 'a')

    def test_unicode_and_multiline_combo_keep_clipboard_strategy(self):
        for text in ('選択', 'first\nsecond', 'x'*321):
            with self.subTest(text=text[:10]):
                element = UIElement(1, 'ComboBoxControl', 'Choice', (10,10,100,30), True, runtime_id=(7,))
                state = ScreenState('', '', [element], 'Form', 100, 100)
                clipboard = Mock()
                clipboard.paste.return_value = 'previous'
                with patch.dict('sys.modules', {'pyperclip': clipboard}), patch('core.controller.time.sleep'), \
                        patch('core.controller.pyautogui') as keyboard, \
                        patch.object(DesktopController, '_typing_focus_matches', return_value=True), \
                        patch.object(DesktopController, '_focused_text', return_value=text):
                    result = DesktopController()._type(NextAction('type', target_tag=1, text=text), state)
                self.assertTrue(result.success)
                keyboard.write.assert_not_called()
                clipboard.copy.assert_any_call(text)
                clipboard.copy.assert_any_call('previous')

    def test_error_after_paste_does_not_replay_keyboard_input(self):
        element = UIElement(1, 'EditControl', 'Text', (10,10,100,30), True, runtime_id=(7,))
        state = ScreenState('', '', [element], 'Form', 100, 100)
        clipboard = Mock()
        clipboard.paste.return_value = 'previous'
        with patch.dict('sys.modules', {'pyperclip': clipboard}), patch('core.controller.time.sleep'), \
                patch('core.controller.pyautogui') as keyboard, \
                patch.object(DesktopController, '_typing_focus_matches', return_value=True), \
                patch.object(DesktopController, '_focused_text', side_effect=RuntimeError('Unavailable')):
            result = DesktopController()._type(NextAction('type', target_tag=1, text='value'), state)
        keyboard.write.assert_not_called()
        self.assertTrue(result.input_unverified)
        self.assertIn('no alternate input method was replayed', result.message)
        clipboard.copy.assert_any_call('previous')

    def type_with_readback(self, values):
        clipboard = Mock()
        clipboard.paste.return_value = 'original clipboard'
        element = UIElement(1, 'EditControl', 'Text input', (10,10,100,30), True, runtime_id=(7,))
        state = ScreenState('', '', [element], 'Form', 100, 100)
        with patch.dict('sys.modules', {'pyperclip': clipboard}), patch('core.controller.time.sleep'), \
                patch('core.controller.pyautogui') as keyboard, \
                patch.object(DesktopController, '_typing_focus_matches', return_value=True), \
                patch.object(DesktopController, '_focused_text', side_effect=values):
            result = DesktopController()._type(NextAction('type', target_tag=1, text='person@example.test'), state)
        clipboard.copy.assert_any_call('original clipboard')
        self.assertEqual(sum(call.args == ('ctrl', 'v') for call in keyboard.hotkey.call_args_list), 1)
        return result

    def test_empty_readback_is_failure_without_paste_replay(self):
        result = self.type_with_readback(['']*5)
        self.assertFalse(result.success)
        self.assertIn('readback mismatch', result.message)

    def test_delayed_readback_waits_without_replaying_input(self):
        result = self.type_with_readback(['', '', 'person@example.test'])
        self.assertTrue(result.success)
        self.assertIn('Autocomplete acceptance and saving are not yet verified', result.message)

    def test_unavailable_readback_is_not_reported_as_success(self):
        result = self.type_with_readback([None]*5)
        self.assertFalse(result.success)
        self.assertIn('readback is unavailable', result.message)

    @patch('core.perception.auto')
    @patch.object(DesktopController, '_typing_focus_matches', return_value=True)
    def test_value_pattern_precedes_empty_text_pattern(self, focus, auto):
        node = auto.GetFocusedControl.return_value
        node.GetValuePattern.return_value.Value = 'person@example.test'
        node.GetTextPattern.return_value.DocumentRange.GetText.return_value = ''
        self.assertEqual(DesktopController._focused_text(self.element()), 'person@example.test')
        node.GetTextPattern.assert_not_called()

    @patch('core.perception.auto')
    def test_editable_child_of_combo_is_valid_focus_but_sibling_is_not(self, auto):
        combo = Mock(ControlTypeName='ComboBoxControl')
        combo.GetRuntimeId.return_value = (7,)
        child = Mock(ControlTypeName='EditControl', IsEnabled=True, IsPassword=False, IsOffscreen=False)
        child.GetRuntimeId.return_value = (8,)
        child.GetParentControl.return_value = combo
        auto.GetFocusedControl.return_value = child
        element = UIElement(1, 'ComboBoxControl', 'Recipient', (10,10,100,30), True, runtime_id=(7,))
        self.assertTrue(DesktopController._typing_focus_matches(element))
        combo.ControlTypeName = 'WindowControl'
        combo.GetRuntimeId.return_value = (9,)
        self.assertFalse(DesktopController._typing_focus_matches(element))
        child.IsPassword = True
        self.assertFalse(DesktopController._typing_focus_matches(element))

    @patch('core.controller.time.sleep')
    @patch('core.controller.pyautogui')
    @patch.object(DesktopController, '_typing_focus_matches', return_value=False)
    def test_click_without_verified_focus_never_selects_or_pastes(self, focus, keyboard, sleep):
        element = UIElement(1, 'ComboBoxControl', 'Recipient', (10,10,100,30), True, runtime_id=(7,))
        state = ScreenState('', '', [element], 'Form', 100, 100)
        result = DesktopController()._type(NextAction('type', target_tag=1, text='person@example.test'), state)
        self.assertFalse(result.success)
        self.assertIn('no text or select-all', result.message)
        keyboard.click.assert_called_once()
        keyboard.hotkey.assert_not_called()
        keyboard.write.assert_not_called()

    def test_visible_combo_retains_runtime_identity(self):
        combo = Node('ComboBoxControl', 'Recipient')
        combo.GetRuntimeId = lambda: (7,)
        with patch('core.perception.UIA_AVAILABLE', True), patch('core.perception.auto') as auto:
            auto.GetForegroundControl.return_value = Node('WindowControl', children=[combo])
            elements, _ = VisualPerceptionEngine().get_accessibility_tree()
        self.assertEqual(elements[0].runtime_id, (7,))

    def element(self):
        return UIElement(1, 'EditControl', 'Untitled-1', (0,0,0,0), True,
                         is_editable=True, runtime_id=(1,2,3), focus_only=True)

    def state(self):
        return ScreenState('', '', [self.element()], 'Untitled-1 - Visual Studio Code', 100,100,42)

    def test_zero_rectangle_editor_is_kept_without_inventing_bounds(self):
        editor = Node('EditControl', 'Untitled-1')
        editor.BoundingRectangle = SimpleNamespace(left=0,top=0,right=0,bottom=0,width=lambda:0,height=lambda:0)
        editor.GetRuntimeId = lambda: (1,2,3)
        with patch('core.perception.UIA_AVAILABLE', True), patch('core.perception.auto') as auto:
            auto.GetForegroundControl.return_value = Node('WindowControl', children=[editor])
            elements,_ = VisualPerceptionEngine().get_accessibility_tree()
            self.assertEqual(len(elements),1)
            self.assertTrue(elements[0].focus_only)
            self.assertEqual(elements[0].bounds,(0,0,0,0))
            editor.IsOffscreen=True
            self.assertEqual(VisualPerceptionEngine().get_accessibility_tree()[0],[])

    @patch('core.perception.auto')
    def test_focus_requires_exact_runtime_identity(self, auto):
        node = Mock(ControlTypeName='EditControl',IsEnabled=True,IsPassword=False,IsOffscreen=False)
        node.GetRuntimeId.return_value = (1,2,3)
        auto.ControlFromHandle.return_value = node
        auto.GetFocusedControl.return_value = node
        self.assertTrue(DesktopController._focus_accessible_editor(self.element(),self.state()))
        node.SetFocus.assert_called_once()
        auto.GetFocusedControl.return_value = Mock()
        auto.GetFocusedControl.return_value.GetRuntimeId.return_value = (9,9)
        self.assertFalse(DesktopController._focus_accessible_editor(self.element(),self.state()))

    @patch('core.controller.pyautogui')
    def test_click_never_uses_zero_coordinates(self, mouse):
        result = DesktopController()._click(NextAction('click',target_tag=1),self.state())
        self.assertFalse(result.success)
        mouse.click.assert_not_called()

    @patch.object(DesktopController,'_focus_accessible_editor',return_value=False)
    @patch('core.controller.pyautogui')
    def test_failed_focus_never_sends_input(self, keyboard, focus):
        result = DesktopController()._type(NextAction('type',target_tag=1,text='sample'),self.state())
        self.assertFalse(result.success)
        keyboard.hotkey.assert_not_called()
        keyboard.click.assert_not_called()

    @patch('core.document_evidence._read',return_value=b'sample')
    def test_saved_editor_named_after_file_can_be_verified(self, read):
        from core.document_evidence import DocumentEvidence
        evidence=DocumentEvidence(r'Save as C:\Test\result.py')
        state=self.state()
        state.elements[0].name='result.py'
        state.elements[0].text='sample'
        self.assertEqual(evidence.check(state),'')


if __name__=='__main__':
    unittest.main()
