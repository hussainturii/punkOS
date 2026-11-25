import unittest
from unittest.mock import Mock, patch
from PIL import Image
from core.perception import ScreenState, UIElement, VisualPerceptionEngine
from core.reasoning import ReasoningCore
from core.sensitive_content import contains_credentials


class SensitiveScreenTests(unittest.TestCase):
    def screen(self, text):
        return ScreenState('synthetic-image', 'synthetic-raw', [UIElement(
            1, 'DocumentControl', 'Text editor', (0, 0, 100, 50), True, text=text,
            is_editable=True)], 'Untitled - Notepad', 100, 100)

    def test_known_credentials_block_all_reasoning_paths(self):
        for text in ('ghp_' + 'x' * 36, '$env:GEMINI_API_KEY = "synthetic-secret-value"'):
            provider = Mock()
            core = ReasoningCore(provider)
            screen = self.screen(text)
            self.assertIsNone(core.plan_task('Open a note', screen))
            self.assertEqual(core.decide_next_action('Open a note', screen, []).raw['blocked_by'], 'sensitive_content')
            self.assertEqual(core.replan_after_failure('Open a note', screen, [], 'retry').raw['blocked_by'], 'sensitive_content')
            provider.call_with_schema.assert_not_called()

    def test_perception_drops_text_and_images_before_dashboard_or_history(self):
        engine = VisualPerceptionEngine()
        screen = self.screen('ghp_' + 'x' * 36)
        with patch.object(engine, 'capture_screen', return_value=Image.new('RGB', (100, 100))), \
             patch.object(engine, 'get_accessibility_tree', return_value=(screen.elements, screen.foreground_window_title)), \
             patch.object(engine, 'img_to_b64') as encode:
            state = engine.perceive()
        self.assertTrue(state.sensitive_content_detected)
        self.assertEqual(state.elements, [])
        self.assertEqual(state.screenshot_b64, '')
        self.assertEqual(state.raw_screenshot_b64, '')
        encode.assert_not_called()

    def test_ordinary_notes_are_not_credentials(self):
        for text in ('Reference: CEDAR-742\nOwner: Nadia\nStatus: ready for review',
                     'API key is available (value hidden).', 'How to configure GEMINI_API_KEY'):
            self.assertFalse(contains_credentials(text))

    def test_hidden_preview_clears_previous_image_without_decoding(self):
        from ui.execution_view import LiveExecutionView
        state = self.screen('ordinary synthetic note')
        state.sensitive_content_detected = True
        state.raw_screenshot_b64 = ''
        view = Mock()
        LiveExecutionView._render_screen(view, state, None)
        self.assertIsNone(view._latest_pixmap)
        view.screen_label.clear.assert_called_once()
        self.assertIn('Preview hidden', view.screen_label.setText.call_args.args[0])


if __name__ == '__main__':
    unittest.main()
