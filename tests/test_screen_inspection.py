import unittest
from unittest.mock import Mock
from core.screen_inspection import inspection_only, inspect_screen
from core.llm import LLMResponse
from core.perception import ScreenState, UIElement


class ScreenInspectionTests(unittest.TestCase):
    def test_browser_inspection_cannot_enter_web_or_desktop_action_loop(self):
        from core.planner import StrategicPlanner
        from core.safety import SafetySystem
        from core.window_handoff import HandoffResult
        perception, reasoning, controller, web = Mock(), Mock(), Mock(), Mock()
        perception.perceive.return_value = self.state
        reasoning.provider = self.provider
        self.respond()
        planner = StrategicPlanner(perception, reasoning, controller, SafetySystem([]),
            web_agent=web, desktop_handoff=lambda _: HandoffResult(True, 'Focused'),
            on_log=lambda _: None)
        result = planner.run('Inspect the Chrome field without typing or clicking. If unreadable, say so.')
        self.assertTrue(result.success)
        controller.execute.assert_not_called()
        web.start.assert_not_called()
        reasoning.plan_task.assert_not_called()
        reasoning.decide_next_action.assert_not_called()

    def setUp(self):
        self.state = ScreenState('annotated-image', 'raw-image', [
            UIElement(1, 'ComboBoxControl', 'Recipient', (0,0,100,20), True, text='Visible text')], 'Form', 100,100)
        self.provider = Mock()
        self.provider.supports_vision.return_value = True
        self.goal = 'Inspect the To field without typing or clicking. If unreadable, say so without guessing.'

    def respond(self, readable=False, findings=None, limitation='Value unreadable', error=None):
        self.provider.call_with_schema.return_value = LLMResponse(dict(readable=readable,
            findings=findings or [], limitation=limitation), '', error)

    def test_explicit_read_only_request_selected_but_creation_not_selected(self):
        self.assertTrue(inspection_only(self.goal))
        self.assertFalse(inspection_only('Create a draft. Inspect it without typing more.'))
        self.assertFalse(inspection_only('Enter an address and report it. Do not send.'))
        self.assertFalse(inspection_only('Read a document'))

    def test_allowed_unreadable_is_completed_inspection_not_verified_value(self):
        self.respond()
        ok, message, history = inspect_screen(self.goal, self.state, self.provider)
        self.assertTrue(ok)
        self.assertIn('unreadable outcome', message)
        self.assertFalse(history[0]['readable'])
        self.assertEqual(self.provider.call_with_schema.call_args.kwargs['image_b64'], 'raw-image')
        self.provider.call_with_schema.assert_called_once()

    def test_unreadable_without_permission_remains_incomplete(self):
        self.respond()
        self.assertFalse(inspect_screen('Inspect the field without typing.', self.state, self.provider)[0])

    def test_provider_error_is_not_converted_into_allowed_unreadable(self):
        self.respond(error='HTTP 503')
        ok, message, _ = inspect_screen(self.goal, self.state, self.provider)
        self.assertFalse(ok)
        self.assertIn('HTTP 503', message)

    def test_screenshot_finding_requires_actual_image(self):
        finding = dict(label='Recipient state', value='Uncommitted text', source='screenshot',
                       evidence='Visible text inside the entry field, no chip shown')
        self.respond(True, [finding], '')
        self.assertTrue(inspect_screen(self.goal, self.state, self.provider)[0])
        self.provider.supports_vision.return_value = False
        self.assertFalse(inspect_screen(self.goal, self.state, self.provider)[0])

    def test_accessibility_quote_must_be_present(self):
        self.respond(True, [dict(label='Field', value='Visible text', source='accessibility', evidence='Invented quote')], '')
        self.assertFalse(inspect_screen(self.goal, self.state, self.provider)[0])

    def test_sensitive_screen_and_cancellation_skip_inference(self):
        self.state.sensitive_content_detected = True
        self.assertFalse(inspect_screen(self.goal, self.state, self.provider)[0])
        self.state.sensitive_content_detected = False
        self.assertFalse(inspect_screen(self.goal, self.state, self.provider, lambda: True)[0])
        self.provider.call_with_schema.assert_not_called()

    def test_empty_readable_report_is_not_success(self):
        self.respond(True, [], '')
        self.assertFalse(inspect_screen(self.goal, self.state, self.provider)[0])
