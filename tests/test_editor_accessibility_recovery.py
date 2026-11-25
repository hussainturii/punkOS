import unittest
from unittest.mock import Mock, patch
import httpx

from core.cloud_llm import CloudJSONProvider
from core.reasoning import ReasoningCore, NEXT_ACTION_SCHEMA
from core.perception import UIElement, ScreenState
from tests.test_cloud_providers import envelope


class EditorAccessibilityTests(unittest.TestCase):
    def test_full_toolbar_after_new_file_is_not_an_editor(self):
        state = self.state()
        state.elements = [UIElement(i, 'ButtonControl', f'Tool {i}', (0, 0, 10, 10), True) for i in range(39)]
        state.elements.append(UIElement(40, 'ButtonControl', 'Screen Reader Optimized', (0, 0, 10, 10), True))
        history = [{'action': "key tag=None text='ctrl+n'", 'success': True}]
        core = ReasoningCore(Mock())
        self.assertEqual(core._perception_blocker(state, history).text, 'alt+f1')
        history.append({'action': "key text='alt+f1'", 'success': True})
        self.assertEqual(core._perception_blocker(state, history).action_type, 'fail')

    def test_malformed_key_retry_is_bounded_without_desktop_input(self):
        from core.planner import StrategicPlanner
        from core.safety import SafetySystem
        perception, reasoning, controller = Mock(), Mock(), Mock()
        perception.perceive.return_value = self.state()
        reasoning.decide_next_action.return_value = ReasoningCore._parse_next_action(
            {'action_type': 'key', 'text': 'print("sample")'})
        planner = StrategicPlanner(perception, reasoning, controller, SafetySystem([]),
            use_planning_phase=False, max_steps=5, step_delay=0, on_log=lambda _: None)
        result = planner.run('Edit the current file')
        self.assertFalse(result.success)
        self.assertEqual(reasoning.decide_next_action.call_count, 2)
        controller.execute.assert_not_called()

    @patch('subprocess.Popen')
    @patch('pathlib.Path.is_file', return_value=True)
    @patch.dict('os.environ', {'LOCALAPPDATA': r'C:\Users\Test\AppData\Local'})
    def test_vscode_launch_requests_renderer_accessibility(self, exists, popen):
        from core.window_handoff import WindowsBackend
        WindowsBackend.__new__(WindowsBackend).launch('vscode')
        self.assertIn('--force-renderer-accessibility', popen.call_args.args[0])
        self.assertFalse(popen.call_args.kwargs['shell'])

    def test_number_shortcuts_and_invalid_syntax(self):
        for combo in ('ctrl+1', 'ctrl+2', '1', 'alt+shift+f1'):
            result = ReasoningCore._parse_next_action({'action_type': 'key', 'text': combo})
            self.assertEqual(result.action_type, 'key')
        for combo in ('', 'ctrl++1', 'press control one'):
            result = ReasoningCore._parse_next_action({'action_type': 'key', 'text': combo, 'reasoning': 'test'})
            self.assertEqual(result.action_type, 'fail')
            self.assertEqual(result.reasoning, 'test')
            self.assertIn('Invalid keyboard action (', result.text)

    def test_sparse_screen_without_document_gets_bounded_recovery(self):
        state = self.state()
        state.elements = state.elements[:1]
        core = ReasoningCore(Mock())
        self.assertEqual(core._perception_blocker(state).text, 'alt+f1')
        for shortcut in ('shift+alt+f1', 'alt+shift+f1'):
            result = core._perception_blocker(state, [{'action': f"key text='{shortcut}'"}])
            self.assertEqual(result.action_type, 'fail')

    def test_help_enable_button_remains_actionable(self):
        state = self.state()
        state.elements.append(UIElement(3, 'ButtonControl', 'Enable screen reader support', (0, 0, 30, 30), True))
        self.assertIsNone(ReasoningCore(Mock())._perception_blocker(state))

    def state(self):
        return ScreenState('', '', [
            UIElement(1, 'ButtonControl', 'Close', (0, 0, 10, 10), True),
            UIElement(2, 'DocumentControl', 'Untitled-1', (0, 0, 100, 100), True, is_interactive=False),
        ], 'Untitled-1 - Visual Studio Code', 100, 100)

    def test_opaque_editor_gets_one_local_recovery_then_actionable_failure(self):
        core = ReasoningCore(Mock())
        state = self.state()
        recovery = core._perception_blocker(state)
        self.assertEqual((recovery.action_type, recovery.text), ('key', 'alt+f1'))
        failed = core._perception_blocker(state, [{'action': "key text='alt+f1'", 'success': True}])
        self.assertEqual(failed.action_type, 'fail')
        self.assertIn('editor remains unobservable', failed.text)
        core.provider.call_with_schema.assert_not_called()

    def test_opaque_editor_cannot_select_type_in_schema(self):
        self.assertNotIn('type', ReasoningCore._action_schema(self.state())['properties']['action_type']['enum'])

    def test_observed_editor_restores_typing(self):
        state = self.state()
        state.elements.append(UIElement(3, 'EditControl', 'Editor content', (0, 0, 100, 100), True, is_editable=True))
        self.assertIsNone(ReasoningCore(Mock())._perception_blocker(state))
        self.assertIn('type', ReasoningCore._action_schema(state)['properties']['action_type']['enum'])
        self.assertIn('[editable: typing allowed]', ReasoningCore._format_elements(state))


class TransientRetryTests(unittest.TestCase):
    def call(self, provider):
        return provider.call_with_schema('', '', NEXT_ACTION_SCHEMA, 'next_action', '')

    @patch('core.cloud_llm.time.sleep')
    @patch('httpx.post')
    def test_503_retries_once_and_logs(self, post, sleep):
        post.side_effect = [httpx.Response(503), httpx.Response(200, json=envelope('gemini'))]
        provider = CloudJSONProvider('gemini', 'fake', 'test-model')
        provider.on_retry = Mock()
        self.assertIsNone(self.call(provider).error)
        self.assertEqual(post.call_count, 2)
        provider.on_retry.assert_called_once()

    @patch('core.cloud_llm.time.sleep')
    @patch('httpx.post', return_value=httpx.Response(503))
    def test_repeated_503_is_bounded(self, post, sleep):
        self.assertIn('HTTP 503', self.call(CloudJSONProvider('gemini', 'fake', 'test-model')).error)
        self.assertEqual(post.call_count, 2)

    @patch('httpx.post')
    def test_quota_auth_and_long_retry_after_are_not_retried(self, post):
        for status, headers in [(429, {}), (401, {}), (503, {'Retry-After': '60'})]:
            post.reset_mock()
            post.return_value = httpx.Response(status, headers=headers)
            self.assertIsNotNone(self.call(CloudJSONProvider('gemini', 'fake', 'test-model')).error)
            self.assertEqual(post.call_count, 1)

    @patch('core.cloud_llm.post_response', return_value=httpx.Response(503))
    def test_cancellation_during_backoff_prevents_second_request(self, post):
        provider = CloudJSONProvider('gemini', 'fake', 'test-model')
        provider.cancel_check = lambda: True
        self.assertIn('cancelled', self.call(provider).error)
        self.assertEqual(post.call_count, 1)


if __name__ == '__main__':
    unittest.main()
