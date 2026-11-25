import unittest
from unittest.mock import patch
from PIL import Image
from core.perception import VisualPerceptionEngine, UIElement


class DialogSettleTests(unittest.TestCase):
    def run_capture(self, snapshots):
        engine = VisualPerceptionEngine()
        with patch.object(engine, 'capture_screen', return_value=Image.new('RGB', (100,100))) as capture, \
             patch.object(engine, 'get_accessibility_tree', side_effect=snapshots) as tree, \
             patch('core.perception.time.sleep') as sleep:
            result = engine.perceive()
        return result, capture.call_count, tree.call_count, sleep.call_count

    def test_sparse_dialog_recaptured_with_new_controls(self):
        close = UIElement(1, 'ButtonControl', 'Close', (0,0,10,10), True)
        field = UIElement(1, 'EditControl', 'File name:', (0,0,20,10), True)
        state, captures, reads, waits = self.run_capture([([close], 'Open'), ([field], 'Open')])
        self.assertEqual(state.elements, [field])
        self.assertEqual((captures, reads, waits), (1,2,1))  # image captured after the final tree read

    def test_retry_is_bounded_when_dialog_stays_empty(self):
        state, captures, reads, waits = self.run_capture([([], 'Open'), ([], 'Open')])
        self.assertEqual((captures, reads, waits), (1,2,1))

    def test_normal_editor_has_no_delay(self):
        _, captures, reads, waits = self.run_capture([([], 'Note - Notepad')])
        self.assertEqual((captures, reads, waits), (1,1,0))

    def test_changed_window_still_runs_sensitive_content_guard(self):
        secret = UIElement(1, 'DocumentControl', 'Text editor', (0,0,10,10), True,
                           text='ghp_' + 'x' * 36)
        state, _, _, _ = self.run_capture([([], 'Open'), ([secret], 'Note - Notepad')])
        self.assertTrue(state.sensitive_content_detected)
        self.assertEqual(state.raw_screenshot_b64, '')


if __name__ == '__main__':
    unittest.main()
