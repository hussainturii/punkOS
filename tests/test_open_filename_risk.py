import unittest
from core.perception import ScreenState, UIElement
from core.reasoning import NextAction
from core.safety import SafetySystem, RiskLevel


class OpenFilenameRiskTests(unittest.TestCase):
    def setUp(self):
        self.safety = SafetySystem([])
        self.state = ScreenState('', '', [UIElement(1, 'EditControl', 'File name:',
                                (0,0,100,30), True)], 'Open', 100, 100)
        self.action = NextAction('type', target_tag=1, text=r'C:\Synthetic\replace-notes\source.txt')

    def test_open_filename_is_not_a_replace_command(self):
        result = self.safety.assess(self.action, screen_state=self.state)
        self.assertEqual(result.level, RiskLevel.LOW)
        self.assertFalse(result.requires_confirmation)

    def test_missing_context_other_fields_and_save_dialog_keep_checks(self):
        self.assertTrue(self.safety.assess(self.action).requires_confirmation)
        for title, name in [('Save As', 'File name:'), ('Open', 'Search Box'), ('Editor', 'File name:')]:
            self.state.foreground_window_title = title
            self.state.elements[0].name = name
            self.assertTrue(self.safety.assess(self.action, screen_state=self.state).requires_confirmation)

    def test_model_high_risk_and_non_path_input_still_require_confirmation(self):
        self.action.is_high_risk = True
        self.assertEqual(self.safety.assess(self.action, screen_state=self.state).level, RiskLevel.HIGH)
        self.action.is_high_risk = False
        self.action.text = 'replace the document'
        self.assertTrue(self.safety.assess(self.action, screen_state=self.state).requires_confirmation)
        self.action.text = 'C:\\replace\\source.txt\nextra command'
        self.assertTrue(self.safety.assess(self.action, screen_state=self.state).requires_confirmation)


if __name__ == '__main__':
    unittest.main()
