"""Regression for document titles containing the agent name."""
import os
import unittest
from unittest.mock import Mock

from core.llm import LLMResponse
from core.perception import ScreenState, UIElement
from core.reasoning import ReasoningCore


class EditorIdentityTests(unittest.TestCase):
    def state(self, title, pid=0):
        return ScreenState('', '', [UIElement(1, 'EditControl', 'Text editor',
            (0, 0, 100, 100), True)], title, 100, 100, foreground_process_id=pid)

    def test_punkos_document_title_is_not_self_window(self):
        for pid in (0, os.getpid() + 1):
            with self.subTest(pid=pid):
                provider = Mock()
                provider.call_with_schema.return_value = LLMResponse(
                    {'action_type': 'type', 'target_tag': 1,
                     'text': 'PunkOS benchmark café\nReady for review.'}, '')
                action = ReasoningCore(provider).decide_next_action(
                    'Type two lines in Notepad', self.state('PunkOS benchmark café - Notepad', pid), [])
                self.assertEqual(action.action_type, 'type')
                self.assertIn('\n', action.text)
                provider.call_with_schema.assert_called_once()

    def test_own_process_is_blocked_in_normal_and_recovery_paths(self):
        provider = Mock()
        core = ReasoningCore(provider)
        state = self.state('Different title', os.getpid())
        for action in (core.decide_next_action('Type a note', state, []),
                       core.replan_after_failure('Type a note', state, [], 'Previous failure')):
            self.assertEqual(action.action_type, 'fail')
            self.assertEqual(action.raw['blocked_by'], 'self_window')
        provider.call_with_schema.assert_not_called()

    def test_exact_legacy_title_remains_blocked_without_pid(self):
        self.assertIsNotNone(ReasoningCore(Mock())._perception_blocker(
            self.state('PunkOS — Agentic Desktop AI')))

    def test_external_process_identity_overrides_title(self):
        self.assertIsNone(ReasoningCore(Mock())._perception_blocker(
            self.state('PunkOS — Agentic Desktop AI', os.getpid() + 1)))


if __name__ == '__main__':
    unittest.main()
