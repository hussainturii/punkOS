import unittest
from unittest.mock import Mock, patch
from core.perception import VisualPerceptionEngine, UIElement, ScreenState
from core.controller import DesktopController
from core.reasoning import NextAction


class EditableDocumentTests(unittest.TestCase):
    def test_explicit_writability_only(self):
        node = Mock(ControlTypeName='DocumentControl', IsPassword=False)
        for value, expected in ((False, True), (0, True), (True, False), (1, False), (None, False), ('false', False)):
            node.GetTextPattern.return_value.DocumentRange.GetAttributeValue.return_value = value
            self.assertEqual(VisualPerceptionEngine._is_editable_document(node), expected)
        node.IsPassword = True
        self.assertFalse(VisualPerceptionEngine._is_editable_document(node))

    @patch('core.controller.time.sleep')
    @patch('pyperclip.copy')
    @patch('pyperclip.paste', return_value='previous clipboard')
    @patch('core.controller.pyautogui')
    def test_verified_editable_document_accepts_literal_unicode(self, keys, paste, copy, sleep):
        editor = UIElement(1, 'DocumentControl', 'Text editor', (0,0,10,10), True, is_editable=True)
        state = ScreenState('', '', [editor], 'Untitled - Notepad', 100, 100)
        text = '  café\nSecond line.\n'
        self.assertTrue(DesktopController().execute(NextAction('type', target_tag=1, text=text), state).success)
        self.assertEqual(copy.call_args_list[0].args, (text,))
        self.assertEqual(copy.call_args_list[-1].args, ('previous clipboard',))
